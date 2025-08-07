// Stock Screener Component with comprehensive signal filtering
import {
  AlertCircle,
  Loader2,
  Minus,
  TrendingDown,
  TrendingUp,
  Wifi,
  WifiOff,
} from "lucide-react";
import React, { useCallback, useEffect, useState } from "react";
import stockScreenerAPI from "../services/api";
import { Badge } from "./ui/badge";
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group";

const Home = () => {
  const [selectedSignal, setSelectedSignal] = useState("all");
  const [stockSignals, setStockSignals] = useState({
    buy_signals: [],
    sell_signals: [],
    hold_signals: [],
  });
  const [holdingsData, setHoldingsData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [websocketConnected, setWebsocketConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Fetch stock signals from API
  const fetchStockSignals = async () => {
    try {
      const data = await stockScreenerAPI.getStockSignals();
      setStockSignals(data);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (error) {
      console.error("Error fetching stock signals:", error);
      setError("Failed to fetch stock signals");
    }
  };

  const fetchHoldingsData = async () => {
    try {
      const data = await stockScreenerAPI.getHoldingsStocks();
      setHoldingsData(data.holdings || []);
    } catch (error) {
      console.error("Error fetching holdings data:", error);
      setError("Failed to fetch holdings data");
    }
  };

  // Fetch system status
  const fetchSystemStatus = async () => {
    try {
      const status = await stockScreenerAPI.getSystemStatus();
      setSystemStatus(status);
    } catch (err) {
      console.error("Error fetching system status:", err);
    }
  };

  // Handle live data updates from WebSocket
  const handleLiveDataUpdate = useCallback((data) => {
    // setLiveData(data); // This line was removed as per the edit hint
    setLastUpdate(new Date().toLocaleTimeString());
  }, []);

  // Handle system status updates from WebSocket
  const handleSystemStatusUpdate = useCallback((data) => {
    setSystemStatus(data);
    setWebsocketConnected(data.websocket_active);
  }, []);

  // Connect to WebSocket for live data
  useEffect(() => {
    // Connect to live data WebSocket
    stockScreenerAPI.connectLiveData(
      handleLiveDataUpdate,
      handleSystemStatusUpdate
    );

    // Set up ping interval to keep connection alive
    const pingInterval = setInterval(() => {
      if (stockScreenerAPI.isWebSocketConnected()) {
        stockScreenerAPI.pingWebSockets();
      }
    }, 30000); // Ping every 30 seconds

    // Set up connection state check
    const connectionCheckInterval = setInterval(() => {
      const connectionState = stockScreenerAPI.getWebSocketState();
      const isConnected = stockScreenerAPI.isWebSocketConnected();

      if (connectionState === "disconnected" && !isConnected) {
        console.log("WebSocket disconnected, attempting to reconnect...");
        stockScreenerAPI.connectLiveData(
          handleLiveDataUpdate,
          handleSystemStatusUpdate
        );
      }
    }, 5000); // Check every 5 seconds

    // Cleanup on unmount
    return () => {
      clearInterval(pingInterval);
      clearInterval(connectionCheckInterval);
      stockScreenerAPI.disconnectWebSockets();
    };
  }, [handleLiveDataUpdate, handleSystemStatusUpdate]);

  // Initial data fetch
  useEffect(() => {
    fetchStockSignals();
    fetchSystemStatus();
    fetchHoldingsData();
  }, []);

  // Auto-refresh every 30 seconds (fallback)
  useEffect(() => {
    const interval = setInterval(() => {
      if (!websocketConnected) {
        fetchStockSignals();
        fetchSystemStatus();
        fetchHoldingsData();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [websocketConnected]);

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
    return signal === "BUY"
      ? "success"
      : signal === "SELL"
      ? "danger"
      : "secondary";
  };

  const getStocksBySignal = () => {
    switch (selectedSignal) {
      case "buy":
        return stockSignals.buy_signals || [];
      case "sell":
        return stockSignals.sell_signals || [];
      case "hold":
        return stockSignals.hold_signals || [];
      case "holdings":
        return holdingsData || [];
      case "all":
        return [
          ...(stockSignals.buy_signals || []),
          ...(stockSignals.sell_signals || []),
          ...(stockSignals.hold_signals || []),
        ];
      default:
        return [
          ...(stockSignals.buy_signals || []),
          ...(stockSignals.sell_signals || []),
          ...(stockSignals.hold_signals || []),
        ];
    }
  };

  const stocks = getStocksBySignal();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
              <p className="text-gray-600">Loading stock signals...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <AlertCircle className="w-8 h-8 mx-auto mb-4 text-red-600" />
              <p className="text-red-600 mb-4">{error}</p>
              <button
                onClick={fetchStockSignals}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

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

          {/* System Status */}
          {systemStatus && (
            <div className="mt-4 p-3 bg-white rounded-lg shadow-sm border">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">System Status:</span>
                <div className="flex items-center space-x-4">
                  <span
                    className={`px-2 py-1 rounded-full text-xs ${
                      systemStatus.zerodha_connected
                        ? "bg-green-100 text-green-800"
                        : "bg-red-100 text-red-800"
                    }`}
                  >
                    {systemStatus.zerodha_connected
                      ? "Connected"
                      : "Disconnected"}
                  </span>
                  <div className="flex items-center space-x-2">
                    {websocketConnected ? (
                      <Wifi className="w-4 h-4 text-green-600" />
                    ) : (
                      <WifiOff className="w-4 h-4 text-red-600" />
                    )}
                    <span
                      className={`px-2 py-1 rounded-full text-xs ${
                        websocketConnected
                          ? "bg-green-100 text-green-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {websocketConnected ? "Live" : "Offline"}
                    </span>
                  </div>
                  <span className="text-gray-500">
                    {systemStatus.stocks_monitored} stocks monitored
                  </span>
                  {lastUpdate && (
                    <span className="text-gray-500 text-xs">
                      Last update: {lastUpdate}
                    </span>
                  )}
                  <span className="text-gray-500 text-xs">
                    Connection: {stockScreenerAPI.getWebSocketState()}
                  </span>
                </div>
              </div>
            </div>
          )}
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
              Buy Signals ({stockSignals.buy_signals?.length || 0})
            </ToggleGroupItem>
            <ToggleGroupItem value="sell" className="px-6 py-2">
              <TrendingDown className="w-4 h-4 mr-2" />
              Sell Signals ({stockSignals.sell_signals?.length || 0})
            </ToggleGroupItem>
            <ToggleGroupItem value="hold" className="px-6 py-2">
              <Minus className="w-4 h-4 mr-2" />
              Hold Signals ({stockSignals.hold_signals?.length || 0})
            </ToggleGroupItem>
            <ToggleGroupItem value="all" className="px-6 py-2">
              <TrendingUp className="w-4 h-4 mr-2" />
              All Signals ({stocks.length || 0})
            </ToggleGroupItem>
            <ToggleGroupItem value="holdings" className="px-6 py-2">
              <TrendingUp className="w-4 h-4 mr-2" />
              Holdings ({holdingsData.length || 0})
            </ToggleGroupItem>
          </ToggleGroup>
        </div>

        {/* Stock Cards Grid */}
        <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
          {/* Table Header */}
          <div className="bg-gray-50 px-6 py-4 border-b">
            <div className="grid grid-cols-12 gap-6 text-sm font-semibold text-gray-700">
              <div className="col-span-1">Section</div>
              <div className="col-span-2">Stock Name</div>
              <div className="col-span-1">Latest Close</div>
              <div className="col-span-1">RSI</div>
              <div className="col-span-1">MA44</div>
              <div className="col-span-1">BB Upper</div>
              <div className="col-span-1">BB Lower</div>
              <div className="col-span-1">Change</div>
              <div className="col-span-1">Volume</div>
              <div className="col-span-1">Latest Update</div>
            </div>
          </div>

          {/* Stock Rows */}
          <div className="divide-y divide-gray-200">
            {stocks.map((stock, index) => (
              <div
                key={index}
                className="px-6 py-4 hover:bg-gray-50 transition-colors"
              >
                <div className="grid grid-cols-12 gap-6 items-center">
                  {/* Section (Buy/Sell/Hold/Holdings) */}
                  <div className="col-span-1">
                    <div
                      className={`px-3 py-1 rounded-full text-xs font-medium text-center ${
                        selectedSignal === "holdings"
                          ? "bg-blue-100 text-blue-800"
                          : stock.signal === "BUY"
                          ? "bg-green-100 text-green-800"
                          : stock.signal === "SELL"
                          ? "bg-red-100 text-red-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {selectedSignal === "holdings"
                        ? "HOLDINGS"
                        : stock.signal}
                    </div>
                  </div>

                  {/* Stock Name and Signal */}
                  <div className="col-span-2">
                    <div className="space-y-1">
                      <div className="font-semibold text-gray-900">
                        {stock.symbol}
                      </div>
                      <Badge
                        variant={getSignalBadgeVariant(stock.signal)}
                        className="text-xs"
                      >
                        {stock.signal}
                      </Badge>
                    </div>
                  </div>

                  {/* Latest Close Price */}
                  <div className="col-span-1">
                    <div className="text-sm font-bold text-gray-900">
                      ₹{stock.close.toLocaleString()}
                    </div>
                  </div>

                  {/* RSI with Direction */}
                  <div className="col-span-1">
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500">RSI</div>
                      <div className="flex items-center space-x-1">
                        <span className="text-sm font-semibold">
                          {stock.indicators.rsi.toFixed(1)}
                        </span>
                        {getDirectionIcon(stock.indicators.rsi_direction)}
                      </div>
                    </div>
                  </div>

                  {/* MA44 with Direction */}
                  <div className="col-span-1">
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500">MA44</div>
                      <div className="flex items-center space-x-1">
                        <span className="text-sm font-semibold">
                          ₹{stock.indicators.ma_44.toLocaleString()}
                        </span>
                        {getDirectionIcon(stock.indicators.ma_44_direction)}
                      </div>
                    </div>
                  </div>

                  {/* BB Upper with Direction */}
                  <div className="col-span-1">
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500">BB Upper</div>
                      <div className="flex items-center space-x-1">
                        <span className="text-sm font-semibold">
                          ₹{stock.indicators.bb_upper.toLocaleString()}
                        </span>
                        {getDirectionIcon(stock.indicators.bb_upper_direction)}
                      </div>
                    </div>
                  </div>

                  {/* BB Lower with Direction */}
                  <div className="col-span-1">
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500">BB Lower</div>
                      <div className="flex items-center space-x-1">
                        <span className="text-sm font-semibold">
                          ₹{stock.indicators.bb_lower.toLocaleString()}
                        </span>
                        {getDirectionIcon(stock.indicators.bb_lower_direction)}
                      </div>
                    </div>
                  </div>

                  {/* Change */}
                  <div className="col-span-1">
                    <div className="space-y-1">
                      <div
                        className={`text-sm font-semibold ${
                          stock.change >= 0 ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        {stock.change >= 0 ? "+" : ""}
                        {stock.change?.toFixed(2) || "0.00"}
                      </div>
                      <div
                        className={`text-xs ${
                          stock.change_percent >= 0
                            ? "text-green-600"
                            : "text-red-600"
                        }`}
                      >
                        ({stock.change_percent?.toFixed(2) || "0.00"}%)
                      </div>
                    </div>
                  </div>

                  {/* Volume */}
                  <div className="col-span-1">
                    <div className="text-sm font-medium text-gray-600">
                      {stock.volume?.toLocaleString() || "N/A"}
                    </div>
                  </div>

                  {/* Latest Update */}
                  <div className="col-span-1">
                    <div className="text-xs text-gray-500">
                      {stock.timestamp
                        ? new Date(stock.timestamp).toLocaleTimeString()
                        : "N/A"}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {stocks.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">
              No {selectedSignal} signals available at the moment.
            </p>
          </div>
        )}

        {/* Refresh Button */}
        <div className="mt-8 text-center">
          <button
            onClick={fetchStockSignals}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Refresh Data
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;
