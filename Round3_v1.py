from math import log, sqrt, exp, pi
from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict, Any, Tuple
from collections import deque
import json

# Product Constants and Parameters
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# Trading parameters for fine-tuning strategies
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "prevent_adverse": False,
        "volume_limit": 0,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.2,
        "min_edge": 2
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.2,
        "min_edge": 2
    },
    Product.PICNIC_BASKET1: {
        "mean": 48.76,
        "std": 47.57,
        "z_entry": 3.0,
        "z_exit": 0.75,
        "max_position": 5,
        "spread_std_window": 45,
    },
    Product.PICNIC_BASKET2: {
        "mean": 30.24,
        "std": 14.93,
        "z_entry": 3.0,
        "z_exit": 0.75,
        "max_position": 5,
        "spread_std_window": 45,
    },
    "BASKET_SPREAD": {
        "mean": 28600,
        "std": 1000,
        "z_entry": 2.5,
        "z_exit": 0.5,
        "max_position": 4,
        "spread_std_window": 45,
    },
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

# Black-Scholes Model for Options Pricing
class BlackScholes:
    @staticmethod
    def norm_cdf(x):
        # Abramowitz and Stegun approximation for erf(z)
        def erf(z):
            if z < 0:
                return -erf(-z)
            a1, a2, a3, a4 = 0.0705230784, 0.0422820123, 0.0092705272, 0.0001520143
            t = 1.0 / (1.0 + a1 * z + a2 * z**2 + a3 * z**3 + a4 * z**4)
            return 1.0 - t**4

        # Normal CDF using erf
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
            return None  # Replace np.nan with None
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

class Trader:
    def __init__(self):
        # Position limits per product
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
        
        # Track market making position windows (for emergency rebalancing)
        self.mm_windows = {
            Product.RAINFOREST_RESIN: deque(maxlen=10),
            Product.KELP: deque(maxlen=10),
            Product.SQUID_INK: deque(maxlen=10),
        }
        
        # Track price history for mean reversion strategies
        self.price_history = {}
        
        # Track spread history for basket strategies
        self.spread_history = {
            Product.PICNIC_BASKET1: [],
            Product.PICNIC_BASKET2: [],
            "BASKET_SPREAD": [],
        }

    # Helper Methods
    def get_position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)
    
    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def get_weighted_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    
    def get_fair_value(self, product: str, order_depth: OrderDepth, trader_data: Dict) -> float:
        if product in PARAMS and "fair_value" in PARAMS[product]:
            return PARAMS[product]["fair_value"]
            
        if product in [Product.KELP, Product.SQUID_INK]:
            if not order_depth.sell_orders or not order_depth.buy_orders:
                return 0
                
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= PARAMS[product]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys() 
                if abs(order_depth.buy_orders[price]) >= PARAMS[product]["adverse_volume"]
            ]
            
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            
            if mm_ask is None or mm_bid is None:
                if f"{product}_last_price" not in trader_data:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = trader_data[f"{product}_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            
            if f"{product}_last_price" in trader_data:
                last_price = trader_data[f"{product}_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * PARAMS[product]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
                
            trader_data[f"{product}_last_price"] = mmmid_price
            return fair
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        popular_bid = max(order_depth.buy_orders.items(), key=lambda x: x[1])[0]
        popular_ask = min(order_depth.sell_orders.items(), key=lambda x: abs(x[1]))[0]
        
        fair_value = (popular_bid + popular_ask) / 2
        return round(fair_value)
    
    def zscore_to_size(self, z: float, max_size: int) -> int:
        capped_z = max(min(abs(z), 5), 1)
        return max(1, int(round((capped_z / 5) * max_size)))

    # Market Making Methods
    def take_best_orders(
        self, 
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0
    ) -> Tuple[List[Order], int, int]:
        position_limit = self.limits[product]
        
        if order_depth.sell_orders:
            sells = sorted(order_depth.sell_orders.items())
            for price, volume in sells:
                if prevent_adverse and abs(volume) <= adverse_volume:
                    continue
                    
                if price <= fair_value - take_width:
                    quantity = min(abs(volume), position_limit - position - buy_order_volume)
                    if quantity > 0:
                        orders.append(Order(product, price, quantity))
                        buy_order_volume += quantity
        
        if order_depth.buy_orders:
            buys = sorted(order_depth.buy_orders.items(), reverse=True)
            for price, volume in buys:
                if prevent_adverse and abs(volume) <= adverse_volume:
                    continue
                    
                if price >= fair_value + take_width:
                    quantity = min(volume, position_limit + position - sell_order_volume)
                    if quantity > 0:
                        orders.append(Order(product, price, -quantity))
                        sell_order_volume += quantity
        
        return orders, buy_order_volume, sell_order_volume
    
    def clear_position_orders(
        self,
        product: str,
        fair_value: float,
        clear_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int
    ) -> Tuple[List[Order], int, int]:
        position_after_trades = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - clear_width)
        fair_for_ask = round(fair_value + clear_width)
        
        buy_quantity = self.limits[product] - (position + buy_order_volume)
        sell_quantity = self.limits[product] + (position - sell_order_volume)
        
        if position_after_trades > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_trades)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        
        if position_after_trades < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_trades))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        
        return orders, buy_order_volume, sell_order_volume
    
    def make_market_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        min_edge: float = 2
    ) -> Tuple[List[Order], int, int]:
        orders = []
        self.mm_windows[product].append(abs(position) >= self.limits[product] * 0.8)
        
        emergency_rebalance = len(self.mm_windows[product]) == 10 and all(self.mm_windows[product])
        risk_off_rebalance = (
            len(self.mm_windows[product]) == 10 and 
            sum(self.mm_windows[product]) >= 5 and 
            self.mm_windows[product][-1]
        )
        
        if product in [Product.KELP, Product.SQUID_INK]:
            aaf = [
                price for price in order_depth.sell_orders.keys()
                if price >= round(fair_value + min_edge)
            ]
            bbf = [
                price for price in order_depth.buy_orders.keys()
                if price <= round(fair_value - min_edge)
            ]
            
            baaf = min(aaf) if aaf else round(fair_value + min_edge)
            bbbf = max(bbf) if bbf else round(fair_value - min_edge)
            
            if emergency_rebalance:
                if position > 0:
                    baaf = min(baaf, round(fair_value))
                elif position < 0:
                    bbbf = max(bbbf, round(fair_value))
                    
            buy_quantity = self.limits[product] - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order(product, round(bbbf + 1), buy_quantity))
                
            sell_quantity = self.limits[product] + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order(product, round(baaf - 1), -sell_quantity))
        else:
            position_adjusted_price = fair_value
            if position > self.limits[product] * 0.5:
                position_adjusted_price -= 2
            elif position < -self.limits[product] * 0.5:
                position_adjusted_price += 2
            
            available_to_buy = self.limits[product] - position - buy_order_volume
            available_to_sell = self.limits[product] + position - sell_order_volume
            
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                
                our_bid = min(best_bid + 1, position_adjusted_price - 1)
                our_ask = max(best_ask - 1, position_adjusted_price + 1)
                
                if emergency_rebalance:
                    if position > 0:
                        our_ask = best_bid
                    elif position < 0:
                        our_bid = best_ask
                elif risk_off_rebalance:
                    if position > 0:
                        our_ask = best_bid + 1
                    elif position < 0:
                        our_bid = best_ask - 1
                
                if available_to_buy > 0:
                    orders.append(Order(product, round(our_bid), available_to_buy))
                if available_to_sell > 0:
                    orders.append(Order(product, round(our_ask), -available_to_sell))
        
        return orders, buy_order_volume, sell_order_volume
    
    def market_making_strategy(
        self, 
        product: str, 
        state: TradingState,
        trader_data: Dict
    ) -> List[Order]:
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        position = self.get_position(state, product)
        
        params = PARAMS.get(product, {})
        take_width = params.get("take_width", 1)
        clear_width = params.get("clear_width", 0.5)
        prevent_adverse = params.get("prevent_adverse", False)
        adverse_volume = params.get("adverse_volume", 0)
        min_edge = params.get("min_edge", 2)
        
        fair_value = self.get_fair_value(product, order_depth, trader_data)
        
        buy_order_volume = 0
        sell_order_volume = 0
        orders, buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume, prevent_adverse, adverse_volume
        )
        
        orders, buy_order_volume, sell_order_volume = self.clear_position_orders(
            product, fair_value, clear_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume
        )
        
        market_orders, _, _ = self.make_market_orders(
            product, order_depth, fair_value, position,
            buy_order_volume, sell_order_volume, min_edge
        )
        orders.extend(market_orders)
        
        return orders
    
    # Basket Trading Methods
    def get_synthetic_basket_depth(
        self, 
        basket_product: str, 
        order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        weights = BASKET_WEIGHTS[basket_product]
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
                
        except (KeyError, ValueError) as e:
            return OrderDepth()
        
        return synthetic_depth
    
    def calculate_basket_spread(
        self,
        basket_product: str,
        order_depths: Dict[str, OrderDepth],
        params: Dict
    ) -> Tuple[float, float]:
        try:
            basket_depth = order_depths[basket_product]
            synthetic_depth = self.get_synthetic_basket_depth(basket_product, order_depths)
            
            if (not basket_depth.buy_orders or not basket_depth.sell_orders or 
                not synthetic_depth.buy_orders or not synthetic_depth.sell_orders):
                return 0, 0
            
            basket_mid = self.get_weighted_mid_price(basket_depth)
            synthetic_mid = self.get_weighted_mid_price(synthetic_depth)
            
            if basket_mid == 0 or synthetic_mid == 0:
                return 0, 0
        
            spread = basket_mid - synthetic_mid
            
            self.spread_history[basket_product].append(spread)
            spread_std_window = params.get("spread_std_window", 45)
            
            if len(self.spread_history[basket_product]) > spread_std_window:
                self.spread_history[basket_product].pop(0)
        
            if len(self.spread_history[basket_product]) >= 5:
                # Custom standard deviation calculation
                values = self.spread_history[basket_product]
                n = len(values)
                if n == 0:
                    rolling_std = 0
                else:
                    mean = sum(values) / n
                    variance = sum((x - mean) ** 2 for x in values) / n
                    rolling_std = sqrt(variance)
                std_to_use = min(rolling_std, params["std"]) if rolling_std > 0 else params["std"]
                z_score = (spread - params["mean"]) / std_to_use if std_to_use > 0 else 0
            else:
                z_score = (spread - params["mean"]) / params["std"] if params["std"] > 0 else 0
        
            return spread, z_score
        
        except (KeyError, ValueError, ZeroDivisionError) as e:
            return 0, 0
    
    def convert_synthetic_basket_orders(
        self,
        basket_product: str,
        synthetic_orders: List[Order],
        order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        component_orders = {}
        for component in BASKET_WEIGHTS[basket_product]:
            component_orders[component] = []
        
        synthetic_basket_depth = self.get_synthetic_basket_depth(basket_product, order_depths)
        best_bid = max(synthetic_basket_depth.buy_orders.keys()) if synthetic_basket_depth.buy_orders else 0
        best_ask = min(synthetic_basket_depth.sell_orders.keys()) if synthetic_basket_depth.sell_orders else float('inf')
        
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            
            try:
                if quantity > 0 and price >= best_ask:
                    component_prices = {}
                    for component in BASKET_WEIGHTS[basket_product]:
                        if not order_depths[component].sell_orders:
                            return {}
                        component_prices[component] = min(order_depths[component].sell_orders.keys())
                
                elif quantity < 0 and price <= best_bid:
                    component_prices = {}
                    for component in BASKET_WEIGHTS[basket_product]:
                        if not order_depths[component].buy_orders:
                            return {}
                        component_prices[component] = max(order_depths[component].buy_orders.keys())
                else:
                    continue
                
                for component, qty in BASKET_WEIGHTS[basket_product].items():
                    component_order = Order(
                        component,
                        component_prices[component],
                        quantity * qty
                    )
                    component_orders[component].append(component_order)
            except (KeyError, ValueError):
                return {}
        
        if all(len(orders) == 0 for orders in component_orders.values()):
            return {}
            
        return component_orders
    
    def execute_basket_orders(
        self,
        target_position: int,
        basket_product: str,
        current_position: int,
        order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        if target_position == current_position:
            return None

        target_quantity = abs(target_position - current_position)
        basket_order_depth = order_depths[basket_product]
        synthetic_order_depth = self.get_synthetic_basket_depth(basket_product, order_depths)

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
                
            basket_orders = [Order(basket_product, basket_ask_price, execute_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_bid_price, -execute_volume)]
            
            component_orders = self.convert_synthetic_basket_orders(
                basket_product, synthetic_orders, order_depths
            )
            component_orders[basket_product] = basket_orders
            return component_orders

        else:
            if not basket_order_depth.buy_orders or not synthetic_order_depth.sell_orders:
                return None
                
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            if execute_volume <= 0:
                return None
                
            basket_orders = [Order(basket_product, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_ask_price, execute_volume)]
            
            component_orders = self.convert_synthetic_basket_orders(
                basket_product, synthetic_orders, order_depths
            )
            component_orders[basket_product] = basket_orders
            return component_orders
    
    def basket_arbitrage_strategy(
        self,
        basket_product: str,
        z_score: float,
        state: TradingState,
        params: Dict
    ) -> Dict[str, List[Order]]:
        weights = BASKET_WEIGHTS[basket_product]
        components = list(weights.keys())
        
        required_products = components + [basket_product]
        if not all(p in state.order_depths for p in required_products):
            return {}
        
        basket_position = self.get_position(state, basket_product)
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
            return self.execute_basket_orders(
                target_position,
                basket_product,
                basket_position,
                state.order_depths
            )
        
        return {}
    
    def calculate_basket_to_basket_spread(
        self,
        order_depths: Dict[str, OrderDepth],
        params: Dict
    ) -> Tuple[float, float]:
        try:
            basket1_depth = order_depths[Product.PICNIC_BASKET1]
            basket2_depth = order_depths[Product.PICNIC_BASKET2]
            
            if (not basket1_depth.buy_orders or not basket1_depth.sell_orders or 
                not basket2_depth.buy_orders or not basket2_depth.sell_orders):
                return 0, 0
            
            basket1_mid = self.get_weighted_mid_price(basket1_depth)
            basket2_mid = self.get_weighted_mid_price(basket2_depth)
            
            spread = basket1_mid - basket2_mid
            
            self.spread_history["BASKET_SPREAD"].append(spread)
            spread_std_window = params.get("spread_std_window", 45)
            
            if len(self.spread_history["BASKET_SPREAD"]) > spread_std_window:
                self.spread_history["BASKET_SPREAD"].pop(0)
            
            if len(self.spread_history["BASKET_SPREAD"]) >= 5:
                # Custom standard deviation calculation
                values = self.spread_history["BASKET_SPREAD"]
                n = len(values)
                if n == 0:
                    rolling_std = 0
                else:
                    mean = sum(values) / n
                    variance = sum((x - mean) ** 2 for x in values) / n
                    rolling_std = sqrt(variance)
                std_to_use = min(rolling_std, params["std"]) if rolling_std > 0 else params["std"]
                z_score = (spread - params["mean"]) / std_to_use if std_to_use > 0 else 0
            else:
                z_score = (spread - params["mean"]) / params["std"] if params["std"] > 0 else 0
            
            return spread, z_score
            
        except (KeyError, ValueError, ZeroDivisionError) as e:
            return 0, 0
    
    def basket_spread_strategy(
        self,
        z_score: float,
        state: TradingState,
        params: Dict
    ) -> Dict[str, List[Order]]:
        result = {}
        
        if not (Product.PICNIC_BASKET1 in state.order_depths and 
                Product.PICNIC_BASKET2 in state.order_depths):
            return result
        
        basket1_position = self.get_position(state, Product.PICNIC_BASKET1)
        basket2_position = self.get_position(state, Product.PICNIC_BASKET2)
        
        basket1_mid = self.get_weighted_mid_price(state.order_depths[Product.PICNIC_BASKET1])
        basket2_mid = self.get_weighted_mid_price(state.order_depths[Product.PICNIC_BASKET2])
        
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
        if basket1_order_qty != 0:
            basket1_price = int(basket1_mid)
            result[Product.PICNIC_BASKET1] = [Order(Product.PICNIC_BASKET1, basket1_price, basket1_order_qty)]
        
        basket2_order_qty = target_basket2_position - basket2_position
        if basket2_order_qty != 0:
            basket2_price = int(basket2_mid)
            result[Product.PICNIC_BASKET2] = [Order(Product.PICNIC_BASKET2, basket2_price, basket2_order_qty)]
        
        return result
    
    # Volcanic Rock Voucher Methods
    def get_volcanic_voucher_mid_price(
        self,
        voucher_order_depth: OrderDepth,
        trader_data: Dict,
        voucher_product: str
    ) -> float:
        if (len(voucher_order_depth.buy_orders) > 0 and len(voucher_order_depth.sell_orders) > 0):
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            trader_data[f"{voucher_product}_prev_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return trader_data.get(f"{voucher_product}_prev_price", 0)

    def delta_hedge_volcanic_rock(
        self,
        rock_order_depth: OrderDepth,
        total_voucher_position: int,
        rock_position: int,
        rock_buy_orders: int,
        rock_sell_orders: int,
        total_delta: float
    ) -> List[Order]:
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

    def volcanic_voucher_orders(
        self,
        voucher_order_depths: Dict[str, OrderDepth],
        voucher_positions: Dict[str, int],
        trader_data: Dict,
        volatilities: Dict[str, float]
    ) -> Dict[str, List[Order]]:
        result = {}
        max_abs_z_score = 0
        selected_voucher = None
        selected_z_score = 0

        # Calculate z-scores for each voucher and find the one with the highest absolute z-score
        for voucher_product in [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]:
            if voucher_product not in trader_data:
                trader_data[voucher_product] = {
                    "prev_price": 0,
                    "past_vol": []
                }

            volatility = volatilities.get(voucher_product, None)
            if volatility is None:
                continue

            trader_data[voucher_product]["past_vol"].append(volatility)
            if len(trader_data[voucher_product]["past_vol"]) < PARAMS[voucher_product]["std_window"]:
                continue
            if len(trader_data[voucher_product]["past_vol"]) > PARAMS[voucher_product]["std_window"]:
                trader_data[voucher_product]["past_vol"].pop(0)

            # Custom standard deviation calculation
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

        # Trade the voucher with the highest absolute z-score if it exceeds the threshold
        if selected_voucher and max_abs_z_score >= PARAMS[selected_voucher]["zscore_threshold"]:
            position = voucher_positions[selected_voucher]
            order_depth = voucher_order_depths[selected_voucher]

            if selected_z_score > 0:
                # Sell voucher (implied volatility too high)
                target_position = -self.limits[selected_voucher]
                if position != target_position:
                    target_quantity = abs(target_position - position)
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        quantity = min(target_quantity, order_depth.buy_orders[best_bid])
                        result[selected_voucher] = [Order(selected_voucher, best_bid, -quantity)]
            else:
                # Buy voucher (implied volatility too low)
                target_position = self.limits[selected_voucher]
                if position != target_position:
                    target_quantity = abs(target_position - position)
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        quantity = min(target_quantity, abs(order_depth.sell_orders[best_ask]))
                        result[selected_voucher] = [Order(selected_voucher, best_ask, quantity)]

        return result

    def volcanic_voucher_strategy(
        self,
        state: TradingState,
        trader_data: Dict
    ) -> Dict[str, List[Order]]:
        result = {}
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result

        rock_mid_price = self.get_mid_price(state.order_depths[Product.VOLCANIC_ROCK])
        if rock_mid_price == 0:
            return result

        # Calculate implied volatility and delta for each voucher
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

            voucher_mid_price = self.get_volcanic_voucher_mid_price(
                state.order_depths[voucher_product],
                trader_data,
                voucher_product
            )
            if voucher_mid_price == 0:
                continue

            tte = PARAMS[voucher_product]["starting_time_to_expiry"] - (state.timestamp / 1_000_000 / 250)
            if tte <= 0:
                tte = 1 / (250 * 100)

            volatility = BlackScholes.implied_volatility(
                voucher_mid_price,
                rock_mid_price,
                PARAMS[voucher_product]["strike"],
                tte
            )
            volatilities[voucher_product] = volatility

            delta = BlackScholes.delta(
                rock_mid_price,
                PARAMS[voucher_product]["strike"],
                tte,
                volatility if volatility is not None else PARAMS[voucher_product]["mean_volatility"]
            )
            total_delta += delta * position

        # Generate voucher orders
        voucher_orders = self.volcanic_voucher_orders(
            state.order_depths,
            voucher_positions,
            trader_data,
            volatilities
        )

        # Delta hedge the total position
        rock_position = self.get_position(state, Product.VOLCANIC_ROCK)
        rock_orders = self.delta_hedge_volcanic_rock(
            state.order_depths[Product.VOLCANIC_ROCK],
            total_voucher_position,
            rock_position,
            0,  # Assuming no other rock orders in this iteration
            0,
            total_delta
        )

        # Combine orders
        for voucher, orders in voucher_orders.items():
            result[voucher] = orders
        if rock_orders:
            result[Product.VOLCANIC_ROCK] = rock_orders

        return result

    # Main Trading Logic
    def run(self, state: TradingState):
        result = {}
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
        
        for product in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK]:
            if product in state.order_depths:
                result[product] = self.market_making_strategy(product, state, trader_data)
        
        if Product.PICNIC_BASKET1 in state.order_depths:
            if not trader_data["spread_history"].get(Product.PICNIC_BASKET1):
                trader_data["spread_history"][Product.PICNIC_BASKET1] = self.spread_history[Product.PICNIC_BASKET1]
            
            spread, z_score = self.calculate_basket_spread(
                Product.PICNIC_BASKET1,
                state.order_depths,
                PARAMS[Product.PICNIC_BASKET1]
            )
            
            self.spread_history[Product.PICNIC_BASKET1] = trader_data["spread_history"][Product.PICNIC_BASKET1]
            
            basket1_orders = self.basket_arbitrage_strategy(
                Product.PICNIC_BASKET1,
                z_score,
                state,
                PARAMS[Product.PICNIC_BASKET1]
            )
            
            if basket1_orders:
                for product, orders in basket1_orders.items():
                    if product not in result:
                        result[product] = orders
                    else:
                        result[product].extend(orders)
        
        if Product.PICNIC_BASKET2 in state.order_depths:
            if not trader_data["spread_history"].get(Product.PICNIC_BASKET2):
                trader_data["spread_history"][Product.PICNIC_BASKET2] = self.spread_history[Product.PICNIC_BASKET2]
            
            spread, z_score = self.calculate_basket_spread(
                Product.PICNIC_BASKET2,
                state.order_depths,
                PARAMS[Product.PICNIC_BASKET2]
            )
            
            self.spread_history[Product.PICNIC_BASKET2] = trader_data["spread_history"][Product.PICNIC_BASKET2]
            
            basket2_orders = self.basket_arbitrage_strategy(
                Product.PICNIC_BASKET2,
                z_score,
                state,
                PARAMS[Product.PICNIC_BASKET2]
            )
            
            if basket2_orders:
                for product, orders in basket2_orders.items():
                    if product not in result:
                        result[product] = orders
                    else:
                        result[product].extend(orders)
        
        if (Product.PICNIC_BASKET1 in state.order_depths and
            Product.PICNIC_BASKET2 in state.order_depths):
            if not trader_data["spread_history"].get("BASKET_SPREAD"):
                trader_data["spread_history"]["BASKET_SPREAD"] = self.spread_history["BASKET_SPREAD"]
            
            spread, z_score = self.calculate_basket_to_basket_spread(
                state.order_depths,
                PARAMS["BASKET_SPREAD"]
            )
            
            self.spread_history["BASKET_SPREAD"] = trader_data["spread_history"]["BASKET_SPREAD"]
            
            basket_spread_orders = self.basket_spread_strategy(
                z_score,
                state,
                PARAMS["BASKET_SPREAD"]
            )
            
            for product, orders in basket_spread_orders.items():
                if product not in result:
                    result[product] = orders
                else:
                    order_by_price = {}
                    for order in result[product]:
                        if order.price not in order_by_price:
                            order_by_price[order.price] = order.quantity
                        else:
                            order_by_price[order.price] += order.quantity
                    
                    for order in orders:
                        if order.price not in order_by_price:
                            order_by_price[order.price] = order.quantity
                        else:
                            order_by_price[order.price] += order.quantity
                    
                    result[product] = [
                        Order(product, price, qty) 
                        for price, qty in order_by_price.items()
                        if qty != 0
                    ]

        # Volcanic Rock Voucher Strategy
        volcanic_orders = self.volcanic_voucher_strategy(state, trader_data)
        for product, orders in volcanic_orders.items():
            if product not in result:
                result[product] = orders
            else:
                result[product].extend(orders)
        
        trader_data["spread_history"] = self.spread_history
        traderData = json.dumps(trader_data)
        
        return result, conversions, traderData