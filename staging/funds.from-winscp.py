<<<<<<< HEAD
# funds/fund_manager.py

class FundManager:
    def __init__(self, initial_fund: float):
        self.total_fund = initial_fund
        self.used_fund = 0.0

    def available_fund(self) -> float:
        return self.total_fund - self.used_fund

    def can_place_order(self, amount: float) -> bool:
        return self.available_fund() >= amount

    def place_order(self, amount: float) -> bool:
        if self.can_place_order(amount):
            self.used_fund += amount
            return True
        return False

    def add_fund(self, amount: float):
        self.total_fund += amount
=======
# funds/fund_manager.py

class FundManager:
    def __init__(self, initial_fund: float):
        self.total_fund = initial_fund
        self.used_fund = 0.0

    def available_fund(self) -> float:
        return self.total_fund - self.used_fund

    def can_place_order(self, amount: float) -> bool:
        return self.available_fund() >= amount

    def place_order(self, amount: float) -> bool:
        if self.can_place_order(amount):
            self.used_fund += amount
            return True
        return False

    def add_fund(self, amount: float):
        self.total_fund += amount
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
# funds/funds.py