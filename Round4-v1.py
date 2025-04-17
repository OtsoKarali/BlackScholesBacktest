from math import log, sqrt, exp, pi
from datamodel import Order, OrderDepth, TradingState, ConversionObservation
from typing import List, Dict, Any, Tuple
from collections import deque
import json
import jsonpickle
import numpy as np

# =============================
# Product Constants
# =============================




# Good
# Macarons : +20,375 PnL

# Mid
# Volcanic Rock : +8000 PnL

# Bad
# 10000 : -1063 PnL
# 10250 : +1188 PnL
# 10500, 750, 9500 : no trades
# Squid_Ink : -451 PnL
# Kelp : +256 PnL
# Resin : +1572 PnL
# Djembes, Jams, Croissants, Baskets : no trades


class Product:
    # Market Making Products
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    # Pairs Trading Products
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
    # Macarons Product
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

# =============================
# Trading Parameters
# =============================

PARAMS = {
    # Volcanic Vouchers
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.154105,
        "strike": 9500,
        "starting_time_to_expiry": 4/250,
        "std_window": 6,
        "zscore_threshold": 1.49,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.161,
        "strike": 9750,
        "starting_time_to_expiry": 4/250,
        "std_window": 6,
        "zscore_threshold": 3.29,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.147766,
        "strike": 10000,
        "starting_time_to_expiry": 4/250,
        "std_window": 6,
        "zscore_threshold": 1.85,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.138391,
        "strike": 10250,
        "starting_time_to_expiry": 4/250,
        "std_window": 6,
        "zscore_threshold": 1.94,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.141255,
        "strike": 10500,
        "starting_time_to_expiry": 4/250,
        "std_window": 6,
        "zscore_threshold": 2.05,
    },
    # Pairs Trading
    Product.PICNIC_BASKET1: {
        "mean": 47.82,
        "std": 8.28,
        "z_entry": 1.6,
        "z_exit": 0.4,
        "max_position": 5,
        "spread_std_window": 45,
    },
    Product.PICNIC_BASKET2: {
        "mean": 29.54,
        "std": 4.92,
        "z_entry": 1.6,
        "z_exit": 0.4,
        "max_position": 5,
        "spread_std_window": 45,
    },
    "BASKET_SPREAD": {
        "mean": 168.28,
        "std": 12.50,
        "z_entry": 1.6,
        "z_exit": 0.4,
        "max_position": 4,
        "spread_std_window": 45,
    },
    # Macarons
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 8.2,
        "make_min_edge": 4.1,
        "make_probability": 0.141,
        "init_make_edge": 8.2,
        "min_edge": 2.1,
        "volume_avg_timestamp": 5,
        "volume_bar": 27,
        "dec_edge_discount": 0.78,
        "step_size": 2.1,
        "position_clear_threshold": 60  # Clear position if it exceeds this
    }
}

# Basket composition weights
BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2
    }
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
# Macarons Strategy
# =============================

class MacaronsStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, params: Dict[str, Any]) -> None:
        super().__init__(symbol, limit)
        self.params = params
        self.price_history = []  # Track prices for volatility calculation
        self.traded_volume_history = []  # Track actual traded volume
        self.conversions_used = 0  # Track conversions per timestamp

    def macarons_implied_bid_ask(self, observation: ConversionObservation) -> tuple[float, float]:
        return (
            observation.bidPrice - observation.exportTariff - observation.transportFees - 0.8,
            observation.askPrice + observation.importTariff + observation.transportFees
        )

    def macarons_adap_edge(
            self, 
            timestamp: int, 
            curr_edge: float, 
            traded_volume: int, 
            traderObject: dict
    ) -> float:
        if timestamp == 0:
            traderObject[self.symbol]["curr_edge"] = self.params[self.symbol]["init_make_edge"]
            return self.params[self.symbol]["init_make_edge"]

        traderObject[self.symbol]["volume_history"].append(traded_volume)
        if len(traderObject[self.symbol]["volume_history"]) > self.params[self.symbol]["volume_avg_timestamp"]:
            traderObject[self.symbol]["volume_history"].pop(0)

        if len(traderObject[self.symbol]["volume_history"]) < self.params[self.symbol]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject[self.symbol]["optimized"]:
            volume_avg = np.mean(traderObject[self.symbol]["volume_history"])

            if volume_avg >= self.params[self.symbol]["volume_bar"]:
                traderObject[self.symbol]["volume_history"] = []
                traderObject[self.symbol]["curr_edge"] = curr_edge + self.params[self.symbol]["step_size"]
                return curr_edge + self.params[self.symbol]["step_size"]

            elif self.params[self.symbol]["dec_edge_discount"] * self.params[self.symbol]["volume_bar"] * (curr_edge - self.params[self.symbol]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[self.symbol]["step_size"] > self.params[self.symbol]["min_edge"]:
                    traderObject[self.symbol]["volume_history"] = []
                    traderObject[self.symbol]["curr_edge"] = curr_edge - self.params[self.symbol]["step_size"]
                    traderObject[self.symbol]["optimized"] = True
                    return curr_edge - self.params[self.symbol]["step_size"]
                else:
                    traderObject[self.symbol]["curr_edge"] = self.params[self.symbol]["min_edge"]
                    return self.params[self.symbol]["min_edge"]

        traderObject[self.symbol]["curr_edge"] = curr_edge
        return curr_edge

    def macarons_clear_position(self, order_depth: OrderDepth, position: int, orders: List[Order], buy_order_volume: int, sell_order_volume: int) -> tuple[List[Order], int, int]:
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders, buy_order_volume, sell_order_volume

        buy_quantity = self.limit - position
        sell_quantity = self.limit + position

        if position > self.params[self.symbol]["position_clear_threshold"]:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items()
                if price >= best_bid - 2
            )
            clear_quantity = min(clear_quantity, position)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(self.symbol, best_bid, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        elif position < -self.params[self.symbol]["position_clear_threshold"]:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items()
                if price <= best_ask + 2
            )
            clear_quantity = min(clear_quantity, abs(position))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(self.symbol, best_ask, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume

    def macarons_arb_take(self, order_depth: OrderDepth, observation: ConversionObservation, adap_edge: float, position: int) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders, buy_order_volume, sell_order_volume

        fair_value = (best_bid + best_ask) / 2

        self.price_history.append(fair_value)
        if len(self.price_history) > 20:
            self.price_history.pop(0)

        volatility = np.std(self.price_history) if len(self.price_history) >= 5 else 0

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        buy_quantity = self.limit - position
        sell_quantity = self.limit + position

        ask = implied_ask + adap_edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - (2.1 + volatility * 0.5)

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[self.symbol]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
                if quantity > 0:
                    orders.append(Order(self.symbol, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
                if quantity > 0:
                    orders.append(Order(self.symbol, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def macarons_arb_clear(self, position: int) -> int:
        conversions = -position
        if abs(conversions) > 10 - self.conversions_used:
            conversions = (10 - self.conversions_used) if conversions > 0 else -(10 - self.conversions_used)
        self.conversions_used += abs(conversions)
        return conversions

    def macarons_arb_make(self, order_depth: OrderDepth, observation: ConversionObservation, position: int, edge: float, buy_order_volume: int, sell_order_volume: int) -> tuple[List[Order], int, int]:
        
        orders: List[Order] = []
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders, buy_order_volume, sell_order_volume

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        volatility = np.std(self.price_history) if len(self.price_history) >= 5 else 0

        position_skew = (position / self.limit) * 2
        bid = implied_bid - edge - position_skew
        ask = implied_ask + edge + position_skew

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - (2.1 + volatility * 0.5)

        if aggressive_ask >= implied_ask + self.params[self.symbol]['min_edge']:
            ask = aggressive_ask
            print("AGGRESSIVE")
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")
        else:
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 9]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 18]

        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        print(f"IMPLIED_BID: {implied_bid}")
        print(f"IMPLIED_ASK: {implied_ask}")
        print(f"FOREIGN ASK: {observation.askPrice}")
        print(f"FOREIGN BID: {observation.bidPrice}")

        buy_quantity = self.limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(self.symbol, round(bid), buy_quantity))

        sell_quantity = self.limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(self.symbol, round(ask), -sell_quantity))

        return orders, buy_order_volume, sell_order_volume

    def act(self, state: TradingState) -> None:
        traderObject = {}
        if state.traderData:
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except:
                traderObject = {}

        if self.symbol not in traderObject:
            traderObject[self.symbol] = {"curr_edge": self.params[self.symbol]["init_make_edge"], "volume_history": [], "optimized": False}

        self.conversions_used = 0  # Reset conversions used per timestamp
        position = state.position.get(self.symbol, 0)
        print(f"{self.symbol} POSITION: {position}")

        if "last_position" in traderObject:
            traded_volume = abs(position - traderObject["last_position"])
        else:
            traded_volume = 0
        traderObject["last_position"] = position
        self.traded_volume_history.append(traded_volume)
        if len(self.traded_volume_history) > 20:
            self.traded_volume_history.pop(0)

        adap_edge = self.macarons_adap_edge(
            state.timestamp,
            traderObject[self.symbol]["curr_edge"],
            traded_volume,
            traderObject,
        )

        take_orders, buy_order_volume, sell_order_volume = self.macarons_arb_take(
            state.order_depths[self.symbol],
            state.observations.conversionObservations[self.symbol],
            adap_edge,
            position,
        )

        clear_orders, buy_order_volume, sell_order_volume = self.macarons_clear_position(
            state.order_depths[self.symbol],
            position,
            take_orders,
            buy_order_volume,
            sell_order_volume
        )

        make_orders, _, _ = self.macarons_arb_make(
            state.order_depths[self.symbol],
            state.observations.conversionObservations[self.symbol],
            position,
            adap_edge,
            buy_order_volume,
            sell_order_volume
        )

        position_after_trades = position + buy_order_volume - sell_order_volume
        if position_after_trades > 0:
            self.conversions = self.macarons_arb_clear(position_after_trades)
        else:
            self.conversions = 0

        self.orders = clear_orders + take_orders + make_orders
        traderObject["traderData"] = jsonpickle.encode(traderObject)
        state.traderData = traderObject["traderData"]

# =============================
# Pairs Trading Strategies
# =============================

class PairsTradingStrategy:
    def __init__(self, basket_product: str, limits: Dict[str, int]) -> None:
        self.basket_product = basket_product
        self.limits = limits
        self.spread_history = []

    def get_weighted_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def zscore_to_size(self, z: float, max_size: int) -> int:
        capped_z = max(min(abs(z), 5), 1)
        return max(1, int(round((capped_z / 5) * max_size)))

    def can_execute_trade(self, state: TradingState, product: str, quantity: int) -> bool:
        current_position = state.position.get(product, 0)
        new_position = current_position + quantity
        return abs(new_position) <= self.limits[product]

    def can_execute_component_trades(self, state: TradingState, basket_product: str, basket_quantity: int) -> bool:
        weights = BASKET_WEIGHTS[basket_product]
        for component, qty in weights.items():
            component_quantity = basket_quantity * qty
            if not self.can_execute_trade(state, component, component_quantity):
                return False
        return True

    def get_synthetic_basket_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        weights = BASKET_WEIGHTS[self.basket_product]
        components = list(weights.keys())
        synthetic_depth = OrderDepth()
        if not all(c in order_depths for c in components):
            return synthetic_depth
        try:
            basket_bid_price = 0
            basket_bid_volume = float('inf')
            if not all(order_depths[component].sell_orders for component in components):
                return synthetic_depth
            for component, qty in weights.items():
                if not order_depths[component].sell_orders:
                    return synthetic_depth
                component_best_ask = min(order_depths[component].sell_orders.keys())
                component_best_ask_vol = abs(order_depths[component].sell_orders[component_best_ask])
                basket_bid_price += component_best_ask * qty
                basket_bid_volume = min(basket_bid_volume, component_best_ask_vol // qty)
            if basket_bid_volume > 0:
                synthetic_depth.buy_orders[basket_bid_price] = basket_bid_volume
            basket_ask_price = 0
            basket_ask_volume = float('inf')
            if not all(order_depths[component].buy_orders for component in components):
                return synthetic_depth
            for component, qty in weights.items():
                if not order_depths[component].buy_orders:
                    return synthetic_depth
                component_best_bid = max(order_depths[component].buy_orders.keys())
                component_best_bid_vol = order_depths[component].buy_orders[component_best_bid]
                basket_ask_price += component_best_bid * qty
                basket_ask_volume = min(basket_ask_volume, component_best_bid_vol // qty)
            if basket_ask_volume > 0:
                synthetic_depth.sell_orders[basket_ask_price] = -basket_ask_volume
        except (KeyError, ValueError):
            return OrderDepth()
        return synthetic_depth

    def calculate_basket_spread(self, order_depths: Dict[str, OrderDepth], params: Dict) -> Tuple[float, float]:
        try:
            basket_depth = order_depths[self.basket_product]
            synthetic_depth = self.get_synthetic_basket_depth(order_depths)
            if (not basket_depth.buy_orders or not basket_depth.sell_orders or 
                not synthetic_depth.buy_orders or not synthetic_depth.sell_orders):
                return 0, 0
            basket_mid = self.get_weighted_mid_price(basket_depth)
            synthetic_mid = self.get_weighted_mid_price(synthetic_depth)
            if basket_mid == 0 or synthetic_mid == 0:
                return 0, 0
            spread = basket_mid - synthetic_mid
            self.spread_history.append(spread)
            spread_std_window = params.get("spread_std_window", 45)
            if len(self.spread_history) > spread_std_window:
                self.spread_history.pop(0)
            if len(self.spread_history) >= 5:
                values = self.spread_history
                n = len(values)
                if n == 0:
                    rolling_std = 0
                else:
                    mean = sum(values) / n
                    variance = sum((x - mean) ** 2 for x in values) / n
                    rolling_std = sqrt(variance)
                std_to_use = rolling_std if rolling_std > 0 else params["std"]
                z_score = (spread - params["mean"]) / std_to_use if std_to_use > 0 else 0
            else:
                z_score = (spread - params["mean"]) / params["std"] if params["std"] > 0 else 0
            return spread, z_score
        except (KeyError, ValueError, ZeroDivisionError):
            return 0, 0

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {component: [] for component in BASKET_WEIGHTS[self.basket_product]}
        synthetic_basket_depth = self.get_synthetic_basket_depth(order_depths)
        best_bid = max(synthetic_basket_depth.buy_orders.keys()) if synthetic_basket_depth.buy_orders else 0
        best_ask = min(synthetic_basket_depth.sell_orders.keys()) if synthetic_basket_depth.sell_orders else float('inf')
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            try:
                if quantity > 0 and price >= best_ask:
                    component_prices = {}
                    for component in BASKET_WEIGHTS[self.basket_product]:
                        if not order_depths[component].sell_orders:
                            return {}
                        component_prices[component] = min(order_depths[component].sell_orders.keys())
                elif quantity < 0 and price <= best_bid:
                    component_prices = {}
                    for component in BASKET_WEIGHTS[self.basket_product]:
                        if not order_depths[component].buy_orders:
                            return {}
                        component_prices[component] = max(order_depths[component].buy_orders.keys())
                else:
                    continue
                for component, qty in BASKET_WEIGHTS[self.basket_product].items():
                    component_order = Order(component, component_prices[component], quantity * qty)
                    component_orders[component].append(component_order)
            except (KeyError, ValueError):
                return {}
        if all(len(orders) == 0 for orders in component_orders.values()):
            return {}
        return component_orders

    def execute_basket_orders(self, target_position: int, current_position: int, order_depths: Dict[str, OrderDepth], state: TradingState) -> Dict[str, List[Order]]:
        if target_position == current_position:
            return None
        target_quantity = target_position - current_position
        if not self.can_execute_trade(state, self.basket_product, target_quantity):
            return None
        if not self.can_execute_component_trades(state, self.basket_product, target_quantity):
            return None
        basket_order_depth = order_depths[self.basket_product]
        synthetic_order_depth = self.get_synthetic_basket_depth(order_depths)
        if target_position > current_position:
            if not basket_order_depth.sell_orders or not synthetic_order_depth.buy_orders:
                return None
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            if execute_volume <= 0:
                return None
            basket_orders = [Order(self.basket_product, basket_ask_price, execute_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_bid_price, -execute_volume)]
            component_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            component_orders[self.basket_product] = basket_orders
            return component_orders
        else:
            if not basket_order_depth.buy_orders or not synthetic_order_depth.sell_orders:
                return None
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, abs(target_quantity))
            if execute_volume <= 0:
                return None
            basket_orders = [Order(self.basket_product, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_ask_price, execute_volume)]
            component_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            component_orders[self.basket_product] = basket_orders
            return component_orders

    def run(self, state: TradingState, params: Dict) -> Dict[str, List[Order]]:
        required_products = list(BASKET_WEIGHTS[self.basket_product].keys()) + [self.basket_product]
        if not all(p in state.order_depths for p in required_products):
            return {}
        basket_position = state.position.get(self.basket_product, 0)
        spread, z_score = self.calculate_basket_spread(state.order_depths, params)
        target_position = 0
        if z_score >= params["z_entry"]:
            target_position = -params["max_position"]
        elif z_score <= -params["z_entry"]:
            target_position = params["max_position"]
        elif abs(z_score) < params["z_exit"]:
            target_position = 0
        else:
            return {}
        if target_position != basket_position:
            orders = self.execute_basket_orders(target_position, basket_position, state.order_depths, state)
            return orders if orders else {}
        return {}

class BasketSpreadStrategy:
    def __init__(self, limits: Dict[str, int]) -> None:
        self.limits = limits
        self.spread_history = []

    def get_weighted_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def zscore_to_size(self, z: float, max_size: int) -> int:
        capped_z = max(min(abs(z), 5), 1)
        return max(1, int(round((capped_z / 5) * max_size)))

    def can_execute_trade(self, state: TradingState, product: str, quantity: int) -> bool:
        current_position = state.position.get(product, 0)
        new_position = current_position + quantity
        return abs(new_position) <= self.limits[product]

    def calculate_basket_to_basket_spread(self, order_depths: Dict[str, OrderDepth], params: Dict) -> Tuple[float, float]:
        try:
            basket1_depth = order_depths[Product.PICNIC_BASKET1]
            basket2_depth = order_depths[Product.PICNIC_BASKET2]
            if (not basket1_depth.buy_orders or not basket1_depth.sell_orders or 
                not basket2_depth.buy_orders or not basket2_depth.sell_orders):
                return 0, 0
            basket1_mid = self.get_weighted_mid_price(basket1_depth)
            basket2_mid = self.get_weighted_mid_price(basket2_depth)
            spread = basket1_mid - basket2_mid
            self.spread_history.append(spread)
            spread_std_window = params.get("spread_std_window", 45)
            if len(self.spread_history) > spread_std_window:
                self.spread_history.pop(0)
            if len(self.spread_history) >= 5:
                values = self.spread_history
                n = len(values)
                if n == 0:
                    rolling_std = 0
                else:
                    mean = sum(values) / n
                    variance = sum((x - mean) ** 2 for x in values) / n
                    rolling_std = sqrt(variance)
                std_to_use = rolling_std if rolling_std > 0 else params["std"]
                z_score = (spread - params["mean"]) / std_to_use if std_to_use > 0 else 0
            else:
                z_score = (spread - params["mean"]) / params["std"] if params["std"] > 0 else 0
            return spread, z_score
        except (KeyError, ValueError, ZeroDivisionError):
            return 0, 0

    def run(self, state: TradingState, params: Dict) -> Dict[str, List[Order]]:
        result = {}
        if not (Product.PICNIC_BASKET1 in state.order_depths and Product.PICNIC_BASKET2 in state.order_depths):
            return result
        basket1_position = state.position.get(Product.PICNIC_BASKET1, 0)
        basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
        basket1_mid = self.get_weighted_mid_price(state.order_depths[Product.PICNIC_BASKET1])
        basket2_mid = self.get_weighted_mid_price(state.order_depths[Product.PICNIC_BASKET2])
        spread, z_score = self.calculate_basket_to_basket_spread(state.order_depths, params)
        target_basket1_position = 0
        target_basket2_position = 0
        position_size = self.zscore_to_size(z_score, params["max_position"])
        if z_score > params["z_entry"]:
            target_basket1_position = -position_size
            target_basket2_position = position_size
        elif z_score < -params["z_entry"]:
            target_basket1_position = position_size
            target_basket2_position = -position_size
        elif abs(z_score) < params["z_exit"]:
            target_basket1_position = 0
            target_basket2_position = 0
        else:
            return result
        basket1_order_qty = target_basket1_position - basket1_position
        basket2_order_qty = target_basket2_position - basket2_position
        if not self.can_execute_trade(state, Product.PICNIC_BASKET1, basket1_order_qty):
            return result
        if not self.can_execute_trade(state, Product.PICNIC_BASKET2, basket2_order_qty):
            return result
        if basket1_order_qty != 0:
            basket1_price = int(basket1_mid)
            result[Product.PICNIC_BASKET1] = [Order(Product.PICNIC_BASKET1, basket1_price, basket1_order_qty)]
        if basket2_order_qty != 0:
            basket2_price = int(basket2_mid)
            result[Product.PICNIC_BASKET2] = [Order(Product.PICNIC_BASKET2, basket2_price, basket2_order_qty)]
        return result

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
            Product.CROISSANTS: 0,
            Product.JAMS: 0,
            Product.DJEMBES: 0,
            Product.PICNIC_BASKET1: 0,
            Product.PICNIC_BASKET2: 0,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.MAGNIFICENT_MACARONS: 75,
        }
        self.strategies = {
            Product.RAINFOREST_RESIN: MarketMakingStrategy(Product.RAINFOREST_RESIN, self.limits[Product.RAINFOREST_RESIN]),
            Product.KELP: MarketMakingStrategy(Product.KELP, self.limits[Product.KELP]),
            Product.SQUID_INK: MarketMakingStrategy(Product.SQUID_INK, self.limits[Product.SQUID_INK]),
            Product.PICNIC_BASKET1: PairsTradingStrategy(Product.PICNIC_BASKET1, self.limits),
            Product.PICNIC_BASKET2: PairsTradingStrategy(Product.PICNIC_BASKET2, self.limits),
            "BASKET_SPREAD": BasketSpreadStrategy(self.limits),
            "VOLCANIC_VOUCHERS": VolcanicVoucherStrategy(self.limits),
            Product.MAGNIFICENT_MACARONS: MacaronsStrategy(Product.MAGNIFICENT_MACARONS, self.limits[Product.MAGNIFICENT_MACARONS], PARAMS),
        }
        self.spread_history = {
            Product.PICNIC_BASKET1: [],
            Product.PICNIC_BASKET2: [],
            "BASKET_SPREAD": [],
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
        if "spread_history" not in trader_data:
            trader_data["spread_history"] = {
                Product.PICNIC_BASKET1: [],
                Product.PICNIC_BASKET2: [],
                "BASKET_SPREAD": [],
            }
        for symbol, strategy in self.strategies.items():
            if symbol in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK, Product.MAGNIFICENT_MACARONS] and symbol in state.order_depths:
                o, c = strategy.run(state)
                orders[symbol] = o
                conversions += c
            elif symbol == Product.PICNIC_BASKET1 and symbol in state.order_depths:
                if not trader_data["spread_history"].get(Product.PICNIC_BASKET1):
                    trader_data["spread_history"][Product.PICNIC_BASKET1] = self.spread_history[Product.PICNIC_BASKET1]
                basket1_orders = strategy.run(state, PARAMS[Product.PICNIC_BASKET1])
                self.spread_history[Product.PICNIC_BASKET1] = trader_data["spread_history"][Product.PICNIC_BASKET1]
                for product, o in basket1_orders.items():
                    if product not in orders:
                        orders[product] = o
                    else:
                        orders[product].extend(o)
            elif symbol == Product.PICNIC_BASKET2 and symbol in state.order_depths:
                if not trader_data["spread_history"].get(Product.PICNIC_BASKET2):
                    trader_data["spread_history"][Product.PICNIC_BASKET2] = self.spread_history[Product.PICNIC_BASKET2]
                basket2_orders = strategy.run(state, PARAMS[Product.PICNIC_BASKET2])
                self.spread_history[Product.PICNIC_BASKET2] = trader_data["spread_history"][Product.PICNIC_BASKET2]
                for product, o in basket2_orders.items():
                    if product not in orders:
                        orders[product] = o
                    else:
                        orders[product].extend(o)
            elif symbol == "BASKET_SPREAD" and Product.PICNIC_BASKET1 in state.order_depths and Product.PICNIC_BASKET2 in state.order_depths:
                if not trader_data["spread_history"].get("BASKET_SPREAD"):
                    trader_data["spread_history"]["BASKET_SPREAD"] = self.spread_history["BASKET_SPREAD"]
                basket_spread_orders = strategy.run(state, PARAMS["BASKET_SPREAD"])
                self.spread_history["BASKET_SPREAD"] = trader_data["spread_history"]["BASKET_SPREAD"]
                for product, o in basket_spread_orders.items():
                    if product not in orders:
                        orders[product] = o
                    else:
                        order_by_price = {}
                        for order in orders[product]:
                            if order.price not in order_by_price:
                                order_by_price[order.price] = order.quantity
                            else:
                                order_by_price[order.price] += order.quantity
                        for order in o:
                            if order.price not in order_by_price:
                                order_by_price[order.price] = order.quantity
                            else:
                                order_by_price[order.price] += order.quantity
                        orders[product] = [Order(product, price, qty) for price, qty in order_by_price.items() if qty != 0]
            elif symbol == "VOLCANIC_VOUCHERS":
                volcanic_orders = strategy.run(state, trader_data)
                for product, o in volcanic_orders.items():
                    if product not in orders:
                        orders[product] = o
                    else:
                        orders[product].extend(o)
        trader_data["spread_history"] = self.spread_history
        traderData = json.dumps(trader_data)
        return orders, conversions, traderData