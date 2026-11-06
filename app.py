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


def review_log(result, uploaded, practice):
    st.subheader('전표별 감사 판단 기록')
    records = st.session_state.setdefault('audit_reviews', {})
    file_id = sha256(uploaded.getvalue()).hexdigest()
    journal_id = st.selectbox('검토 전표', result.Journal_ID.tolist(), key=f'review_{file_id}')
    key = (file_id, journal_id)
    previous = records.get(key, {})
    levels = ['우선 검토', '추가 확인', '낮은 우선순위']
    with st.form(f'review_form_{file_id}_{journal_id}'):
        level = st.selectbox('내 검토 우선순위', levels,
                             index=levels.index(previous.get('내 우선순위', levels[0])))
        reason = st.text_area('내 판단 근거', previous.get('내 판단 근거', ''))
        evidence = st.text_area('확인할 증빙과 판단을 바꿀 조건', previous.get('증빙과 판단 변경 조건', ''))
        difference = ''
        if previous or not practice:
            difference = st.text_area('자동 탐지와 내 판단의 차이', previous.get('탐지 결과와의 차이', ''))
        saved = st.form_submit_button('판단 저장')
    if saved:
        if not reason.strip() or not evidence.strip():
            st.warning('판단 근거와 확인할 증빙을 입력하세요.')
        else:
            row = result.loc[result.Journal_ID.eq(journal_id)].iloc[0]
            records[key] = {
                '원본 파일': uploaded.name, '파일 식별값': file_id, '전표번호': journal_id,
                '내 우선순위': level, '내 판단 근거': reason.strip(),
                '자동 탐지 사유': row['Anomaly Reason'] or '탐지 없음',
                '자동 탐지 점수': int(row['Risk Score']),
                '탐지 결과와의 차이': difference.strip(),
                '증빙과 판단 변경 조건': evidence.strip(),
            }
            st.rerun()
    if previous:
        st.write(f"저장된 자동 탐지 결과 · {previous['자동 탐지 사유']} ({previous['자동 탐지 점수']}점)")
    if records:
        history = pd.DataFrame(records.values())
        st.dataframe(history, hide_index=True)
        st.download_button('감사 판단 기록 CSV 다운로드',
                           history.to_csv(index=False).encode('utf-8-sig'),
                           file_name='gl_audit_review_log.csv', mime='text/csv')
    st.caption('기록은 현재 세션에 유지됩니다. 이후에도 보관하려면 CSV를 내려받으세요.')


def main():
    st.set_page_config(page_title='GL Audit Dashboard', page_icon='🔎', layout='wide')
    st.title('GL Audit Dashboard')
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
    review_log(result, uploaded, practice)
    if practice:
        return
    risky = flags.any(axis=1)
    a, b = st.columns(2)
    a.metric('전체 전표 수', f'{len(df):,}')
    b.metric('위험 전표 수 · 중복 제외', f'{int(risky.sum()):,}',
             help='하나의 전표에서 여러 위험이 발견되어도 한 건으로 집계합니다.')
    for card, name in zip(st.columns(5), flags.columns):
        card.metric(name, f'{int(flags[name].sum()):,}')
    st.caption('카드는 업로드 전체 기준입니다. 유형별 건수는 중복될 수 있습니다.')
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
