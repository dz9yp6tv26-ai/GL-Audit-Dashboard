import io
import pandas as pd
import streamlit as st

COLUMNS = ['Journal_ID', 'Date', 'Debit_Account', 'Debit_Amount',
           'Credit_Account', 'Credit_Amount', 'Preparer_ID', 'Approver_ID', 'Description']


def load_csv(raw, encoding='utf-8-sig'):
    df = pd.read_csv(io.BytesIO(raw), encoding=encoding, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError('필수 컬럼 누락: ' + ', '.join(sorted(missing)))
    df = df[COLUMNS].copy()
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
        raise ValueError('차변과 대변 금액이 다른 전표가 있습니다. 원본을 확인하세요.')
    return df


def detect(df, lower, upper, revenue_accounts, allowed_debits):
    dt = df.Date.dt
    amount = df.Debit_Amount
    # 야간: 22:00 이상 ~ 다음날 06:00 미만. 날짜는 업로드한 현지 시각 기준.
    flags = pd.DataFrame({
        '비업무시간': (dt.dayofweek >= 5) | (dt.hour >= 22) | (dt.hour < 6),
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
    return result, flags


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
    risky = flags.any(axis=1)
    a, b = st.columns(2)
    a.metric('전체 전표 수', f'{len(df):,}')
    b.metric('위험 전표 수 · 중복 제외', f'{int(risky.sum()):,}',
             help='하나의 전표에서 여러 위험이 발견되어도 한 건으로 집계합니다.')
    for card, name in zip(st.columns(5), flags.columns):
        card.metric(name, f'{int(flags[name].sum()):,}')
    st.caption('카드는 업로드 전체 기준입니다. 유형별 건수는 중복될 수 있습니다.')
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
               '기말은 매년 12/25~12/31입니다. 임계치 구간은 쪼개기 의심 후보이며 '
               '계정 조합은 설정한 매출 대변 규칙만 검사합니다. 탐지는 감사 검토 대상 선별입니다.')


if __name__ == '__main__':
    main()
