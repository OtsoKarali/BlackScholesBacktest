from math import log, sqrt, exp, pi
from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict, Any, Tuple
from collections import deque
import json

# =============================
# Product Constants
# =============================

class Product:
    # Market Making Products (Original)
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    # Basket Market Making Products
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    # Volcanic Voucher Products
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# =============================
# Trading Parameters
# =============================

PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.154105,
        "strike": 9500,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 1.49,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.161,
        "strike": 9750,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 3.29,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.147766,
        "strike": 10000,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 1.85,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.138391,
        "strike": 10250,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 1.94,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.141255,
        "strike": 10500,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 2.05,
    },
}

# =============================
# Shared Base Strategy Class
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

# =============================
# Market Making Strategy
# =============================

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10
        self.true_value = None
        if symbol == Product.RAINFOREST_RESIN:
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

# =============================
# Black-Scholes Model
# =============================

class BlackScholes:
    @staticmethod
    def norm_cdf(x):
        def erf(z):
            if z < 0:
                return -erf(-z)
            a1, a2, a3, a4 = 0.0705230784, 0.0422820123, 0.0092705272, 0.0001520143
            t = 1.0 / (1.0 + a1 * z + a2 * z**2 + a3 * z**3 + a4 * z**4)
            return 1.0 - t**4
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0:
            return 0
        d1 = (log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * BlackScholes.norm_cdf(d1) - strike * BlackScholes.norm_cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0:
            return 0
        d1 = (log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return BlackScholes.norm_cdf(d1)

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        if call_price <= 0 or spot <= 0 or time_to_expiry <= 0:
            return None
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            if estimated_price is None:
                return None
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                return volatility
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

# =============================
# Volcanic Voucher Strategy
# =============================

class VolcanicVoucherStrategy:
    def __init__(self, limits: Dict[str, int]) -> None:
        self.limits = limits

    def get_position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def get_volcanic_voucher_mid_price(self, voucher_order_depth: OrderDepth, trader_data: Dict, voucher_product: str) -> float:
        if len(voucher_order_depth.buy_orders) > 0 and len(voucher_order_depth.sell_orders) > 0:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            trader_data[f"{voucher_product}_prev_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return trader_data.get(f"{voucher_product}_prev_price", 0)

    def delta_hedge_volcanic_rock(self, rock_order_depth: OrderDepth, total_voucher_position: int, rock_position: int, rock_buy_orders: int, rock_sell_orders: int, total_delta: float) -> List[Order]:
        target_rock_position = -int(total_delta * total_voucher_position)
        hedge_quantity = target_rock_position - (rock_position + rock_buy_orders - rock_sell_orders)
        orders: List[Order] = []
        if hedge_quantity > 0:
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(abs(hedge_quantity), -rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.limits[Product.VOLCANIC_ROCK] - (rock_position + rock_buy_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif hedge_quantity < 0:
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(abs(hedge_quantity), rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.limits[Product.VOLCANIC_ROCK] + (rock_position - rock_sell_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))
        return orders

    def volcanic_voucher_orders(self, voucher_order_depths: Dict[str, OrderDepth], voucher_positions: Dict[str, int], trader_data: Dict, volatilities: Dict[str, float]) -> Dict[str, List[Order]]:
        result = {}
        max_abs_z_score = 0
        selected_voucher = None
        selected_z_score = 0
        for voucher_product in [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]:
            if voucher_product not in trader_data:
                trader_data[voucher_product] = {"prev_price": 0, "past_vol": []}
            volatility = volatilities.get(voucher_product, None)
            if volatility is None:
                continue
            trader_data[voucher_product]["past_vol"].append(volatility)
            if len(trader_data[voucher_product]["past_vol"]) < PARAMS[voucher_product]["std_window"]:
                continue
            if len(trader_data[voucher_product]["past_vol"]) > PARAMS[voucher_product]["std_window"]:
                trader_data[voucher_product]["past_vol"].pop(0)
            values = trader_data[voucher_product]["past_vol"]
            n = len(values)
            if n == 0:
                vol_std = 0
            else:
                mean = sum(values) / n
                variance = sum((x - mean) ** 2 for x in values) / n
                vol_std = sqrt(variance)
            if vol_std == 0:
                continue
            z_score = (volatility - PARAMS[voucher_product]["mean_volatility"]) / vol_std
            abs_z_score = abs(z_score)
            if abs_z_score > max_abs_z_score:
                max_abs_z_score = abs_z_score
                selected_voucher = voucher_product
                selected_z_score = z_score
        if selected_voucher and max_abs_z_score >= PARAMS[selected_voucher]["zscore_threshold"]:
            position = voucher_positions[selected_voucher]
            order_depth = voucher_order_depths[selected_voucher]
            if selected_z_score > 0:
                target_position = -self.limits[selected_voucher]
                if position != target_position:
                    target_quantity = abs(target_position - position)
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        quantity = min(target_quantity, order_depth.buy_orders[best_bid])
                        result[selected_voucher] = [Order(selected_voucher, best_bid, -quantity)]
            else:
                target_position = self.limits[selected_voucher]
                if position != target_position:
                    target_quantity = abs(target_position - position)
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        quantity = min(target_quantity, abs(order_depth.sell_orders[best_ask]))
                        result[selected_voucher] = [Order(selected_voucher, best_ask, quantity)]
        return result

    def run(self, state: TradingState, trader_data: Dict) -> Dict[str, List[Order]]:
        result = {}
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result
        rock_mid_price = self.get_mid_price(state.order_depths[Product.VOLCANIC_ROCK])
        if rock_mid_price == 0:
            return result
        voucher_positions = {}
        volatilities = {}
        total_delta = 0
        total_voucher_position = 0
        for voucher_product in [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]:
            if voucher_product not in state.order_depths:
                continue
            position = self.get_position(state, voucher_product)
            voucher_positions[voucher_product] = position
            total_voucher_position += position
            voucher_mid_price = self.get_volcanic_voucher_mid_price(state.order_depths[voucher_product], trader_data, voucher_product)
            if voucher_mid_price == 0:
                continue
            tte = PARAMS[voucher_product]["starting_time_to_expiry"] - (state.timestamp / 1_000_000 / 250)
            if tte <= 0:
                tte = 1 / (250 * 100)
            volatility = BlackScholes.implied_volatility(voucher_mid_price, rock_mid_price, PARAMS[voucher_product]["strike"], tte)
            volatilities[voucher_product] = volatility
            delta = BlackScholes.delta(rock_mid_price, PARAMS[voucher_product]["strike"], tte, volatility if volatility is not None else PARAMS[voucher_product]["mean_volatility"])
            total_delta += delta * position
        voucher_orders = self.volcanic_voucher_orders(state.order_depths, voucher_positions, trader_data, volatilities)
        rock_position = self.get_position(state, Product.VOLCANIC_ROCK)
        rock_orders = self.delta_hedge_volcanic_rock(state.order_depths[Product.VOLCANIC_ROCK], total_voucher_position, rock_position, 0, 0, total_delta)
        for voucher, orders in voucher_orders.items():
            result[voucher] = orders
        if rock_orders:
            result[Product.VOLCANIC_ROCK] = rock_orders
        return result

# =========================
# Main Trader Entry Point
# =========================

class Trader:
    def __init__(self):
        self.limits = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
        }
        self.strategies = {
            Product.RAINFOREST_RESIN: MarketMakingStrategy(Product.RAINFOREST_RESIN, self.limits[Product.RAINFOREST_RESIN]),
            Product.KELP: MarketMakingStrategy(Product.KELP, self.limits[Product.KELP]),
            Product.SQUID_INK: MarketMakingStrategy(Product.SQUID_INK, self.limits[Product.SQUID_INK]),
            Product.CROISSANTS: MarketMakingStrategy(Product.CROISSANTS, self.limits[Product.CROISSANTS]),
            Product.JAMS: MarketMakingStrategy(Product.JAMS, self.limits[Product.JAMS]),
            Product.DJEMBES: MarketMakingStrategy(Product.DJEMBES, self.limits[Product.DJEMBES]),
            Product.PICNIC_BASKET1: MarketMakingStrategy(Product.PICNIC_BASKET1, self.limits[Product.PICNIC_BASKET1]),
            Product.PICNIC_BASKET2: MarketMakingStrategy(Product.PICNIC_BASKET2, self.limits[Product.PICNIC_BASKET2]),
            "VOLCANIC_VOUCHERS": VolcanicVoucherStrategy(self.limits),
        }

    def run(self, state: TradingState):
        orders = {}
        conversions = 0
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except:
                trader_data = {}
        for symbol, strategy in self.strategies.items():
            if symbol in [
                Product.RAINFOREST_RESIN,
                Product.KELP,
                Product.SQUID_INK,
                Product.CROISSANTS,
                Product.JAMS,
                Product.DJEMBES,
                Product.PICNIC_BASKET1,
                Product.PICNIC_BASKET2
            ] and symbol in state.order_depths:
                o, c = strategy.run(state)
                orders[symbol] = o
                conversions += c
            elif symbol == "VOLCANIC_VOUCHERS":
                volcanic_orders = strategy.run(state, trader_data)
                for product, o in volcanic_orders.items():
                    if product not in orders:
                        orders[product] = o
                    else:
                        orders[product].extend(o)
        traderData = json.dumps(trader_data)
        return orders, conversions, traderData