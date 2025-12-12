from funds import FundManager
import os
fn = os.getenv("FUND_STATE_FILE", "funds_state_test.json")
fm = FundManager(state_file=fn)
print("before:", fm, fm.available_fund())
fm.add_funds(20000)
<<<<<<< HEAD
print("after:", fm, fm.available_fund())
=======
print("after:", fm, fm.available_fund())
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
