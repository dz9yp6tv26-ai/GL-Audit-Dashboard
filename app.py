import io
from hashlib import sha256
import pandas as pd
import streamlit as st

COLUMNS = ['Journal_ID', 'Date', 'Debit_Account', 'Debit_Amount',
           'Credit_Account', 'Credit_Amount', 'Preparer_ID', 'Approver_ID', 'Description']
SOURCE_COLUMNS = ['Source', 'Entry_Type', 'Journal_Source']


def load_csv(raw, encoding='utf-8-sig'):
    df = pd.read_csv(io.BytesIO(raw), encoding=encoding, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError('필수 컬럼 누락: ' + ', '.join(sorted(missing)))
    source_col = next((col for col in SOURCE_COLUMNS if col in df.columns), None)
    df = df[COLUMNS + ([source_col] if source_col else [])].copy()
    if df.empty:
        raise ValueError('데이터 행이 없습니다.')
    for col in COLUMNS:
        df[col] = df[col].str.strip()
    for col in set(COLUMNS) - {'Description'}:
        if df[col].eq('').any():
            raise ValueError(f'{col}: 빈 값이 있습니다.')
    if df.Journal_ID.duplicated().any():
        raise ValueError('Journal_ID가 중복됩니다. 이 앱은 한 행당 한 전표를 전제로 합니다.')
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d %H:%M', errors='coerce')
    if df.Date.isna().any():
        raise ValueError('Date는 YYYY-MM-DD HH:MM 형식이어야 합니다.')
    for col in ['Debit_Amount', 'Credit_Amount']:
        # 원화 정수만 허용하여 라운드 금액 검사를 정수 연산으로 수행합니다.
        values = df[col].str.replace(',', '', regex=False)
        if not values.str.fullmatch(r'\d{1,18}').all():
            raise ValueError(f'{col}: 0 이상 원화 정수(최대 18자리)를 입력하세요.')
        df[col] = values.astype('int64')
    if df.Debit_Amount.ne(df.Credit_Amount).any():
        bad = df.index[df.Debit_Amount.ne(df.Credit_Amount)].tolist()
        details = ', '.join(
            f'{df.loc[i, "Journal_ID"]}(차변 {df.loc[i, "Debit_Amount"]:,}원 / 대변 {df.loc[i, "Credit_Amount"]:,}원)'
            for i in bad[:5]
        )
        suffix = f' 외 {len(bad) - 5}건' if len(bad) > 5 else ''
        raise ValueError(f'차대변 금액 불일치 {len(bad)}건: {details}{suffix}. 데이터 정정 후 다시 업로드하세요.')
    return df


def detect(df, lower, upper, revenue_accounts, allowed_debits):
    dt = df.Date.dt
    amount = df.Debit_Amount
    # 야간: 22:00 이상 ~ 다음날 06:00 미만. 날짜는 업로드한 현지 시각 기준.
    off_hours = (dt.dayofweek >= 5) | (dt.hour >= 22) | (dt.hour < 6)
    source_col = next((col for col in SOURCE_COLUMNS if col in df.columns), None)
    if source_col:
        system_generated = df[source_col].str.strip().str.casefold().isin(
            {'system generated', 'system_generated', 'system', '자동', '자동생성', '시스템 생성', '시스템'}
        )
        off_hours &= ~system_generated
    flags = pd.DataFrame({
        '비업무시간': off_hours,
        '임계치 구간': amount.between(lower, upper),
        '직무분리 위반': df.Preparer_ID.str.casefold().eq(df.Approver_ID.str.casefold()),
        '기말 라운드 금액': (dt.month.eq(12) & dt.day.between(25, 31)
                         & amount.ge(100_000_000) & amount.mod(1_000_000).eq(0)),
        '계정 조합': (df.Credit_Account.isin(revenue_accounts)
                    & ~df.Debit_Account.isin(allowed_debits)),
    }, index=df.index)
    result = df.copy()
    result['Anomaly Reason'] = flags.apply(
        lambda row: '; '.join(row.index[row].tolist()), axis=1)
    weights = {'비업무시간': 20, '임계치 구간': 15, '직무분리 위반': 30,
               '기말 라운드 금액': 25, '계정 조합': 25}
    result['Risk Score'] = flags.mul(pd.Series(weights)).sum(axis=1).clip(upper=100).astype(int)
    return result, flags


REVIEW_OPTIONS = {
    '비업무시간': (
        ['ERP 입력·승인 로그', '배치 실행 로그', '거래처 현지시각 기록'],
        ['승인된 정기 배치와 일치하면 우선순위를 낮춤', '승인 이력이 없으면 우선순위를 높임'],
    ),
    '임계치 구간': (
        ['작업지시서와 관련 발주서', '거래처 청구서와 지급 이력', '승인 기준과 승인 로그'],
        ['별도 작업과 독립 증빙이 확인되면 우선순위를 낮춤', '동일 작업 합계가 기준 이상이고 추가 승인이 없으면 우선순위를 높임'],
    ),
    '직무분리 위반': (
        ['ERP 승인 이력', '사용자 권한표와 대체 승인 기록'],
        ['독립 검토 기록이 확인되면 우선순위를 낮춤', '작성자가 단독으로 승인했으면 우선순위를 높임'],
    ),
    '기말 라운드 금액': (
        ['결산 산출표와 근거 자료', '환입 전표와 후속 확정 청구서'],
        ['금액을 재계산할 수 있고 후속 증빙과 일치하면 우선순위를 낮춤', '금액 산출 근거가 없으면 우선순위를 높임'],
    ),
    '계정 조합': (
        ['계약서와 인도·검수 기록', '관련 보조원장과 분개 승인 기록'],
        ['거래 실질과 계정 사용 근거가 일치하면 우선순위를 낮춤', '계정 분류나 매출 인식 시점이 증빙과 불일치하면 우선순위를 높임'],
    ),
}


def review_log(result, flags, uploaded):
    st.subheader('전표별 감사 판단 기록')
    records = st.session_state.setdefault('audit_reviews', {})
    file_id = sha256(uploaded.getvalue()).hexdigest()
    journal_id = st.selectbox('검토 전표', result.Journal_ID.tolist(), key=f'review_{file_id}')
    key = (file_id, journal_id)
    previous = records.get(key, {})
    levels = ['우선 검토', '추가 확인', '낮은 우선순위']
    widget_key = f'audit_{file_id}_{journal_id}'
    level = st.selectbox('내 검토 우선순위', levels, key=f'{widget_key}_level',
                         index=levels.index(previous.get('내 우선순위', levels[0])))
    risks = st.multiselect('내 판단 근거 · 위험지표', list(flags.columns),
                           default=previous.get('내 위험지표', '').splitlines(),
                           key=f'{widget_key}_risks',
                           help='해당하는 위험지표가 없으면 비워 두세요.')
    choices = {}
    for index, field in enumerate(['확인할 증빙', '판단을 바꿀 조건']):
        options = [f'{risk} · {option}' for risk in risks for option in REVIEW_OPTIONS[risk][index]]
        field_key = f'{widget_key}_{index}'
        current = st.session_state.get(field_key, previous.get(field, '').splitlines())
        valid = [choice for choice in current if choice in options]
        st.session_state[field_key] = valid
        choices[field] = st.multiselect(field, options, key=field_key, disabled=not options)
    memo = st.text_area('메모 (선택)', previous.get('메모', ''), key=f'{widget_key}_memo')
    if st.button('판단 저장', key=f'{widget_key}_save'):
        row = result.loc[result.Journal_ID.eq(journal_id)].iloc[0]
        auto = flags.loc[row.name]
        mine_only = [risk for risk in risks if not auto[risk]]
        auto_only = [risk for risk in flags.columns if auto[risk] and risk not in risks]
        matched = len(risks) - len(mine_only)
        compared = len(risks) + len(auto_only)
        score = round(100 * matched / compared) if compared else 100
        records[key] = {
            '전표번호': journal_id, '자동 탐지 기준 일치율 (%)': score,
            '내 우선순위': level, '내 위험지표': '\n'.join(risks),
            **{field: '\n'.join(values) for field, values in choices.items()},
            '자동 탐지 사유': row['Anomaly Reason'] or '탐지 없음',
            '내 판단에만 있음': ', '.join(mine_only) or '없음',
            '자동 탐지에만 있음': ', '.join(auto_only) or '없음',
            '메모': memo.strip(),
        }
        previous = records[key]
        st.success('판단을 저장했습니다.')
    if previous:
        st.markdown('#### 채점 결과')
        mine = set(previous.get('내 위험지표', '').splitlines())
        detected = set(previous['자동 탐지 사유'].split('; ')) - {'탐지 없음'}
        compared = len(mine | detected)
        matched = len(mine & detected)
        score = round(100 * matched / compared) if compared else 100
        st.metric('자동 탐지 기준 일치율', f'{score}%')
        st.caption(f'비교 대상 {compared}개 중 {matched}개 일치' if compared
                   else '내 선택과 자동 탐지 모두 위험지표 없음')
        st.write('**일치한 지표** · ' + (', '.join(sorted(mine & detected)) or '없음'))
        st.write('**내가 추가로 선택한 지표** · ' + previous['내 판단에만 있음'])
        st.write('**자동 탐지에만 나온 지표** · ' + previous['자동 탐지에만 있음'])
        st.caption('일치율은 앱의 탐지 규칙과 비교한 점수입니다. 증빙과 메모는 검토 기록에 남습니다.')
    current_records = [record for (source, _), record in records.items() if source == file_id]
    if current_records:
        history = pd.DataFrame(current_records).drop(
            columns=['원본 파일', '파일 식별값', '자동 탐지 점수'], errors='ignore')
        st.dataframe(history, hide_index=True)
        st.download_button('감사 판단 기록 CSV 다운로드',
                           history.to_csv(index=False).encode('utf-8-sig'),
                           file_name='gl_audit_review_log.csv', mime='text/csv')
    st.caption('기록은 현재 세션에 유지됩니다. 이후에도 보관하려면 CSV를 내려받으세요.')


def main():
    title = 'K-GAAS 부정위험요인 기반 분개장(JE) 자동 검증 대시보드'
    st.set_page_config(page_title=title, page_icon='🔎', layout='wide')
    st.markdown("""
        <style>
        .stMainBlockContainer {padding-top: 0; padding-bottom: 4rem;}
        [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stSidebar"] {
            background: #f0ebfa;
        }
        [data-testid="stHeader"] {
            background: transparent;
            color: #fff;
        }
        .je-hero {
            background: #57368c;
            box-shadow: 0 0 0 100vmax #57368c;
            clip-path: inset(0 -100vmax);
            padding: clamp(3.75rem, 7vw, 5rem) 0 clamp(1.5rem, 3vw, 2.5rem);
        }
        .je-hero h1 {
            color: #fff; font-size: clamp(1.45rem, 3vw, 2.5rem);
            line-height: 1.35; letter-spacing: -.035em; padding: 0; margin: 0;
        }
        .je-hero p {
            color: #ded3ee; font-size: clamp(.8rem, 1.4vw, .98rem);
            line-height: 1.55; font-weight: 600; margin: .65rem 0 0;
        }
        [data-testid="stMetric"] {
            background: var(--st-secondary-background-color, #f5f4ff);
            border: 1px solid #7778b539; border-radius: 16px;
            padding: 1rem; box-shadow: 0 4px 14px #393d810a;
        }
        [data-testid="stMetricValue"] {font-weight: 700; font-variant-numeric: tabular-nums;}
        [data-testid="stFileUploaderDropzone"] {border: 1px dashed #8587c9; border-radius: 16px;}
        [data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
        </style>
        <header class="je-hero">
          <h1>K-GAAS 부정위험요인 기반 분개장(JE) 자동 검증 대시보드</h1>
          <p>Automated JE Scanner: K-GAAS Fraud Risk &amp; Anomaly Detection System</p>
        </header>
        """, unsafe_allow_html=True)
    st.caption('분개장 테스트 · 원화 기준 · 한 행당 차변/대변 한 쌍의 전표')
    with st.sidebar:
        st.header('탐지 설정')
        encoding = st.selectbox('CSV 인코딩', ['utf-8-sig', 'cp949'])
        lower = st.number_input('임계치 하한 (원 · 포함)', min_value=0, value=9_000_000, step=100_000)
        upper = st.number_input('임계치 상한 (원 · 포함)', min_value=0, value=9_999_999, step=100_000)
        revenue = st.text_area('대변 매출 계정 (한 줄에 하나)', '제품매출\n상품매출\n매출액')
        allowed = st.text_area('허용 차변 계정 (한 줄에 하나)',
                               '매출채권\n현금\n보통예금\n현금및현금성자산\n현금 및 현금성자산')
        st.caption('계정명은 앞뒤 공백 제거 후 정확히 대조합니다. 회사 계정 체계에 맞게 수정하세요.')
    if lower > upper:
        st.error('임계치 하한은 상한 이하여야 합니다.')
        return
    revenue_accounts = {s.strip() for s in revenue.splitlines() if s.strip()}
    allowed_debits = {s.strip() for s in allowed.splitlines() if s.strip()}
    if not revenue_accounts or not allowed_debits:
        st.error('매출 계정과 허용 차변 계정을 각각 하나 이상 입력하세요.')
        return
    uploaded = st.file_uploader('총계정원장 CSV 업로드', type=['csv'])
    if uploaded is None:
        st.info('앞서 생성한 CSV를 저장한 뒤 업로드하세요. 금액에는 통화 기호를 넣지 마세요.')
        return
    try:
        df = load_csv(uploaded.getvalue(), encoding)
    except (ValueError, UnicodeError, pd.errors.ParserError, OverflowError) as exc:
        st.error(f'CSV 검증 실패: {exc}')
        return
    result, flags = detect(df, lower, upper, revenue_accounts, allowed_debits)
    practice = st.checkbox('실습 모드 · 내 판단을 저장한 전표의 탐지 결과만 확인')
    if practice:
        review_log(result, flags, uploaded)
        return
    risky = flags.any(axis=1)
    st.subheader('전표 검토 개요')
    a, b, c = st.columns(3)
    a.metric('전체 전표 수', f'{len(df):,}')
    b.metric('위험 전표 수 · 중복 제외', f'{int(risky.sum()):,}',
             help='하나의 전표에서 여러 위험이 발견되어도 한 건으로 집계합니다.')
    c.metric('탐지 전표 비율', f'{risky.mean():.1%}')
    st.markdown('**위험지표별 탐지 건수**')
    st.bar_chart(flags.sum().rename_axis('위험지표').reset_index(name='전표 수'),
                 x='위험지표', y='전표 수', color='#595bc3')
    st.caption('업로드 전체 기준이며, 한 전표가 여러 위험지표에 집계될 수 있습니다.')
    with st.expander('전표별 감사 판단 기록', expanded=True):
        review_log(result, flags, uploaded)
    st.subheader('감사 우선 검토 · High Risk 상위 10건')
    priority = result.loc[flags.any(axis=1)].sort_values(
        ['Risk Score', 'Journal_ID'], ascending=[False, True], kind='stable'
    ).head(10).copy()
    if priority.empty:
        st.info('검토 우선순위 대상 전표가 없습니다.')
    else:
        priority['Date'] = priority.Date.dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(priority, hide_index=True, use_container_width=True)
    st.caption('위험 점수는 비업무시간 20점, 임계치 15점, 직무분리 위반 30점, '
               '기말 라운드 25점, 계정 조합 25점의 합계이며 최대 100점입니다. '
               '감사 검토 순서 선정을 돕는 지표로, 부정 여부를 확정하지 않습니다.')
    st.subheader('탐지 전표')
    selected = st.multiselect('위험 유형 필터 · 선택 유형 중 하나라도 충족',
                              flags.columns.tolist(), default=flags.columns.tolist())
    show_all = st.checkbox('정상 전표도 함께 표시', value=False)
    mask = pd.Series(True, index=df.index) if show_all else flags[selected].any(axis=1)
    view = result.loc[mask].copy()
    view['Date'] = view.Date.dt.strftime('%Y-%m-%d %H:%M')
    st.write(f'표시 전표: {len(view):,}건')
    if view.empty:
        st.info('선택 조건에 해당하는 전표가 없습니다.')
    else:
        def highlight(row):
            color = 'background-color: #fff1cc; color: #352a0b' if row['Anomaly Reason'] else ''
            return [color] * len(row)
        st.dataframe(view.style.apply(highlight, axis=1), hide_index=True, use_container_width=True)
    st.download_button('현재 조회 결과 CSV 다운로드',
                       view.to_csv(index=False).encode('utf-8-sig'),
                       file_name='gl_audit_results.csv', mime='text/csv')
    st.caption('야간은 22:00 이상~06:00 미만이며 주말을 포함합니다. '
               'Source, Entry_Type 또는 Journal_Source 열에서 system generated/자동생성으로 표시한 전표는 비업무시간 탐지에서 제외합니다. '
               '기말은 매년 12/25~12/31입니다. 임계치 구간은 쪼개기 의심 후보이며 '
               '계정 조합은 설정한 매출 대변 규칙만 검사합니다. 탐지는 감사 검토 대상 선별입니다.')


if __name__ == '__main__':
    main()
