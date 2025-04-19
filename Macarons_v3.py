class MacaronsStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, params: Dict[str, Any]) -> None:
        super().__init__(symbol, limit)
        self.params = params
        self.price_history = []
        self.traded_volume_history = []
        self.conversions_used = 0
        self.CSI = 49.75  # Critical Sunlight Index from the graph
        self.sunlight_history = []  # To dynamically adjust CSI if needed

    def get_sunlight_index(self, observation: ConversionObservation) -> float:
        return observation.sunlightIndex

    def macarons_adap_edge(
            self, 
            timestamp: int, 
            curr_edge: float, 
            traded_volume: int, 
            traderObject: dict,
            sunlight_index: float
    ) -> float:
        if timestamp == 0:
            traderObject[self.symbol]["curr_edge"] = self.params[self.symbol]["init_make_edge"]
            return self.params[self.symbol]["init_make_edge"]

        traderObject[self.symbol]["volume_history"].append(traded_volume)
        if len(traderObject[self.symbol]["volume_history"]) > self.params[self.symbol]["volume_avg_timestamp"]:
            traderObject[self.symbol]["volume_history"].pop(0)

        if len(traderObject[self.symbol]["volume_history"]) < self.params[self.symbol]["volume_avg_timestamp"]:
            return curr_edge

        # Adjust edge based on sunlight index deviation from CSI
        deviation = (self.CSI - sunlight_index) / self.CSI  # Normalized deviation
        if sunlight_index < self.CSI:
            # Increase edge proportionally to the deviation (more aggressive when sunlight is lower)
            edge_multiplier = 1 + max(0, deviation * 0.5)  # Up to 50% increase
            curr_edge *= edge_multiplier
        else:
            # Reduce edge when above CSI, but less aggressively
            edge_multiplier = 1 - min(0.2, deviation * 0.3)  # Up to 20% decrease
            curr_edge *= edge_multiplier

        if not traderObject[self.symbol]["optimized"]:
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

    def macarons_clear_position(self, order_depth: OrderDepth, position: int, orders: List[Order], buy_order_volume: int, sell_order_volume: int, sunlight_index: float) -> tuple[List[Order], int, int]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders, buy_order_volume, sell_order_volume

        buy_quantity = self.limit - position
        sell_quantity = self.limit + position

        # Adjust position clearing threshold dynamically
        position_clear_threshold = self.params[self.symbol]["position_clear_threshold"]
        if sunlight_index < self.CSI:
            deviation = (self.CSI - sunlight_index) / self.CSI
            position_clear_threshold *= (1 + deviation)  # Allow larger positions when sunlight is lower
        else:
            position_clear_threshold *= 0.8  # Tighten positions when sunlight is above CSI

        if position > position_clear_threshold:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items()
                if price >= best_bid - 2
            )
            clear_quantity = min(clear_quantity, position)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(self.symbol, best_bid, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        elif position < -position_clear_threshold:
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

    def macarons_arb_take(self, order_depth: OrderDepth, observation: ConversionObservation, adap_edge: float, position: int, sunlight_index: float) -> tuple[List[Order], int, int]:
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

        # Bias towards buying when sunlight is below CSI
        if sunlight_index < self.CSI:
            for price in sorted(list(order_depth.sell_orders.keys())):
                if price > implied_bid - edge:
                    break
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
                if quantity > 0:
                    orders.append(Order(self.symbol, round(price), quantity))
                    buy_order_volume += quantity
        else:
            # Standard arbitrage behavior when above CSI
            for price in sorted(list(order_depth.sell_orders.keys())):
                if price > implied_bid - edge:
                    break
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
                if quantity > 0:
                    orders.append(Order(self.symbol, round(price), quantity))
                    buy_order_volume += quantity

            for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
                if price < implied_ask + edge:
                    break
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
                if quantity > 0:
                    orders.append(Order(self.symbol, round(price), -quantity))
                    sell_order_volume += quantity

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

        self.conversions_used = 0
        position = state.position.get(self.symbol, 0)
        print(f"{self.symbol} POSITION: {position}")

        # Get sunlight index
        observation = state.observations.conversionObservations[self.symbol]
        sunlight_index = self.get_sunlight_index(observation)
        print(f"SUNLIGHT INDEX: {sunlight_index}")

        if "last_position" in traderObject:
            traded_volume = abs(position - traderObject["last_position"])
        else:
            traded_volume = 0
        traderObject["last_position"] = position
        self.traded_volume_history.append(traded_volume)
        if len(self.traded_volume_history) > 20:
            self.traded_volume_history.pop(0)

        # Pass sunlight index to adap_edge
        adap_edge = self.macarons_adap_edge(
            state.timestamp,
            traderObject[self.symbol]["curr_edge"],
            traded_volume,
            traderObject,
            sunlight_index
        )

        # Pass sunlight index to arb_take
        take_orders, buy_order_volume, sell_order_volume = self.macarons_arb_take(
            state.order_depths[self.symbol],
            observation,
            adap_edge,
            position,
            sunlight_index
        )

        # Pass sunlight index to clear_position
        clear_orders, buy_order_volume, sell_order_volume = self.macarons_clear_position(
            state.order_depths[self.symbol],
            position,
            take_orders,
            buy_order_volume,
            sell_order_volume,
            sunlight_index
        )

        make_orders, _, _ = self.macarons_arb_make(
            state.order_depths[self.symbol],
            observation,
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