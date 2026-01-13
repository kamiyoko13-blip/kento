from funds import FundManager
import os
fn = os.getenv("FUND_STATE_FILE", "funds_state_test.json")
fm = FundManager(state_file=fn)
print("before:", fm, fm.available_fund())
fm.add_funds(20000)
print("after:", fm, fm.available_fund())
