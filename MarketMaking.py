from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
from collections import deque
import json

# =============================
# Shared Base Strategy Classes
# =============================

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders = []
        self.conversions = 0

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0
        self.act(state)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass

    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

# ===========================================
# Market Making Strategy
# ===========================================

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10
        self.true_value = None
        if symbol == "RAINFOREST_RESIN":
            self.true_value = 10000

    def get_fair_value(self, state: TradingState) -> int:
        if self.true_value is not None:
            return self.true_value

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        if not buy_orders or not sell_orders:
            return 0

        popular_buy = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell = min(sell_orders, key=lambda tup: tup[1])[0]
        return round((popular_buy + popular_sell) / 2)

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        true_value = self.get_fair_value(state)
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        emergency_rebalance = len(self.window) == self.window_size and all(self.window)
        risk_off_rebalance = (
            len(self.window) == self.window_size
            and sum(self.window) >= self.window_size / 2
            and self.window[-1]
        )

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < -self.limit * 0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and emergency_rebalance:
            self.buy(true_value, to_buy // 2)
            to_buy -= to_buy // 2
        if to_buy > 0 and risk_off_rebalance:
            self.buy(true_value - 2, to_buy // 2)
            to_buy -= to_buy // 2
        if to_buy > 0 and buy_orders:
            popular_bid = max(buy_orders, key=lambda tup: tup[1])[0]
            bid_price = min(max_buy_price, popular_bid + 1)
            self.buy(bid_price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and emergency_rebalance:
            self.sell(true_value, to_sell // 2)
            to_sell -= to_sell // 2
        if to_sell > 0 and risk_off_rebalance:
            self.sell(true_value + 2, to_sell // 2)
            to_sell -= to_sell // 2
        if to_sell > 0 and sell_orders:
            popular_ask = min(sell_orders, key=lambda tup: tup[1])[0]
            ask_price = max(min_sell_price, popular_ask - 1)
            self.sell(ask_price, to_sell)

# =========================
# Main Trader Entry Point
# =========================

class Trader:
    def __init__(self):
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }

        self.strategies = {
            "RAINFOREST_RESIN": MarketMakingStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": MarketMakingStrategy("KELP", limits["KELP"]),
            "SQUID_INK": MarketMakingStrategy("SQUID_INK", limits["SQUID_INK"]),
        }

    def run(self, state: TradingState):
        orders = {}
        conversions = 0
        traderData = ""

        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths:
                o, c = strategy.run(state)
                orders[symbol] = o
                conversions += c

        return orders, conversions, traderData