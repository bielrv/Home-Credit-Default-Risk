# Checking bureau data

Select distinct 
b.CREDIT_ACTIVE
,count(distinct b.SK_ID_CURR)
,count(*) from bureau b
group by CREDIT_ACTIVE;

Select distinct 
b.CREDIT_CURRENCY
,count(distinct b.SK_ID_CURR)
,count(*) from bureau b
group by CREDIT_CURRENCY;

Select distinct 
b.CREDIT_TYPE
,count(distinct b.SK_ID_CURR)
,count(*) from bureau b
group by CREDIT_TYPE;

# Checking bureau balance data

Select distinct bb.`STATUS`
,count(distinct bb.SK_ID_BUREAU)
,count(bb.MONTHS_BALANCE)
from bureau_balance bb
group by bb.`STATUS`;

# Checking previous_application balance data


