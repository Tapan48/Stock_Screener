import { Minus, TrendingDown, TrendingUp } from "lucide-react";
import React, { useState } from "react";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group";

const Home = () => {
  const [selectedSignal, setSelectedSignal] = useState("buy");

  // Mock data for demonstration
  const buyStocks = [
    {
      name: "RELIANCE",
      signal: "BUY",
      close: 2456.8,
      rsi: 32.5,
      rsiDirection: "ASCENDING",
      ma44: 2420.5,
      ma44Direction: "ASCENDING",
      bbUpper: 2480.2,
      bbUpperDirection: "ASCENDING",
      bbLower: 2380.1,
      bbLowerDirection: "ASCENDING",
    },
    {
      name: "TCS",
      signal: "BUY",
      close: 3890.45,
      rsi: 28.2,
      rsiDirection: "ASCENDING",
      ma44: 3850.3,
      ma44Direction: "CONSTANT",
      bbUpper: 3920.8,
      bbUpperDirection: "ASCENDING",
      bbLower: 3780.2,
      bbLowerDirection: "ASCENDING",
    },
    {
      name: "INFY",
      signal: "BUY",
      close: 1567.9,
      rsi: 35.8,
      rsiDirection: "ASCENDING",
      ma44: 1550.4,
      ma44Direction: "ASCENDING",
      bbUpper: 1580.6,
      bbUpperDirection: "ASCENDING",
      bbLower: 1520.3,
      bbLowerDirection: "ASCENDING",
    },
  ];

  const sellStocks = [
    {
      name: "HDFCBANK",
      signal: "SELL",
      close: 1650.75,
      rsi: 72.3,
      rsiDirection: "DESCENDING",
      ma44: 1680.2,
      ma44Direction: "DESCENDING",
      bbUpper: 1690.4,
      bbUpperDirection: "DESCENDING",
      bbLower: 1620.1,
      bbLowerDirection: "DESCENDING",
    },
    {
      name: "ICICIBANK",
      signal: "SELL",
      close: 890.45,
      rsi: 68.9,
      rsiDirection: "DESCENDING",
      ma44: 910.3,
      ma44Direction: "CONSTANT",
      bbUpper: 920.8,
      bbUpperDirection: "DESCENDING",
      bbLower: 880.2,
      bbLowerDirection: "DESCENDING",
    },
  ];

  const getDirectionIcon = (direction) => {
    switch (direction) {
      case "ASCENDING":
        return <TrendingUp className="w-4 h-4 text-green-600" />;
      case "DESCENDING":
        return <TrendingDown className="w-4 h-4 text-red-600" />;
      case "CONSTANT":
        return <Minus className="w-4 h-4 text-gray-600" />;
      default:
        return null;
    }
  };

  const getSignalBadgeVariant = (signal) => {
    return signal === "BUY" ? "success" : "danger";
  };

  const stocks = selectedSignal === "buy" ? buyStocks : sellStocks;

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Stock Screener
          </h1>
          <p className="text-gray-600">
            Platform to simplify stock analysis and generate trading signals
          </p>
        </div>

        {/* Toggle Bar */}
        <div className="mb-8">
          <ToggleGroup
            type="single"
            value={selectedSignal}
            onValueChange={(value) => value && setSelectedSignal(value)}
            className="bg-white shadow-sm border rounded-lg p-1"
          >
            <ToggleGroupItem value="buy" className="px-6 py-2">
              <TrendingUp className="w-4 h-4 mr-2" />
              Buy Signals
            </ToggleGroupItem>
            <ToggleGroupItem value="sell" className="px-6 py-2">
              <TrendingDown className="w-4 h-4 mr-2" />
              Sell Signals
            </ToggleGroupItem>
          </ToggleGroup>
        </div>

        {/* Stock Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {stocks.map((stock, index) => (
            <Card
              key={index}
              className="bg-white shadow-sm hover:shadow-md transition-shadow"
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xl">{stock.name}</CardTitle>
                  <Badge variant={getSignalBadgeVariant(stock.signal)}>
                    {stock.signal}
                  </Badge>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  ₹{stock.close.toLocaleString()}
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {/* RSI */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">
                      RSI
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-semibold">{stock.rsi}</span>
                      {getDirectionIcon(stock.rsiDirection)}
                    </div>
                  </div>

                  {/* MA44 */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">
                      MA44
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-semibold">
                        ₹{stock.ma44.toLocaleString()}
                      </span>
                      {getDirectionIcon(stock.ma44Direction)}
                    </div>
                  </div>

                  {/* Bollinger Bands */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-600">
                        BB Upper
                      </span>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-semibold">
                          ₹{stock.bbUpper.toLocaleString()}
                        </span>
                        {getDirectionIcon(stock.bbUpperDirection)}
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-600">
                        BB Lower
                      </span>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-semibold">
                          ₹{stock.bbLower.toLocaleString()}
                        </span>
                        {getDirectionIcon(stock.bbLowerDirection)}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {stocks.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">
              No {selectedSignal} signals available at the moment.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;
