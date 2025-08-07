import {
  AlertCircle,
  ArrowLeft,
  Loader2,
  Minus,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import React, { useEffect, useState } from "react";
import stockScreenerAPI from "../services/api";

const StockDetail = ({ symbol, onBack }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stockData, setStockData] = useState(null);
  const [stockHistory, setStockHistory] = useState([]);
  const [selectedDays, setSelectedDays] = useState("30");

  useEffect(() => {
    const loadStockData = async () => {
      try {
        setLoading(true);
        const [details, history] = await Promise.all([
          stockScreenerAPI.getStockDetails(symbol),
          stockScreenerAPI.getStockHistory(symbol, parseInt(selectedDays)),
        ]);
        setStockData(details);
        setStockHistory(history.history || []);
      } catch (error) {
        console.error("Error loading stock data:", error);
        setError("Failed to load stock data");
      } finally {
        setLoading(false);
      }
    };

    loadStockData();
  }, [symbol, selectedDays]);

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

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
              <p className="text-gray-600">Loading stock data...</p>
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
                onClick={onBack}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Back to List
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
          <button
            onClick={onBack}
            className="flex items-center text-gray-600 hover:text-gray-900 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to List
          </button>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            {symbol} Details
          </h1>
          {stockData && <p className="text-gray-600">{stockData.name}</p>}
        </div>

        {/* Stock Overview Card */}
        {stockData && (
          <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
            <div className="grid grid-cols-4 gap-8">
              {/* Latest Price */}
              <div className="text-center">
                <h3 className="text-sm font-medium text-gray-500 mb-2">
                  Latest Price
                </h3>
                <p className="text-2xl font-bold text-gray-900">
                  ₹{stockData.close.toLocaleString()}
                </p>
              </div>

              {/* RSI */}
              <div className="text-center">
                <h3 className="text-sm font-medium text-gray-500 mb-2">RSI</h3>
                <div className="flex items-center justify-center">
                  <p className="text-2xl font-bold text-gray-900 mr-2">
                    {stockData.indicators.rsi.toFixed(2)}
                  </p>
                  {getDirectionIcon(stockData.indicators.rsi_direction)}
                </div>
              </div>

              {/* MA44 */}
              <div className="text-center">
                <h3 className="text-sm font-medium text-gray-500 mb-2">MA44</h3>
                <div className="flex items-center justify-center">
                  <p className="text-2xl font-bold text-gray-900 mr-2">
                    ₹{stockData.indicators.ma_44.toLocaleString()}
                  </p>
                  {getDirectionIcon(stockData.indicators.ma_44_direction)}
                </div>
              </div>

              {/* Bollinger Bands */}
              <div className="text-center">
                <h3 className="text-sm font-medium text-gray-500 mb-2">
                  Bollinger Bands
                </h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-center">
                    <span className="text-sm text-gray-500 mr-2">Upper:</span>
                    <span className="text-lg font-medium">
                      ₹{stockData.indicators.bb_upper.toLocaleString()}
                    </span>
                    {getDirectionIcon(stockData.indicators.bb_upper_direction)}
                  </div>
                  <div className="flex items-center justify-center">
                    <span className="text-sm text-gray-500 mr-2">Lower:</span>
                    <span className="text-lg font-medium">
                      ₹{stockData.indicators.bb_lower.toLocaleString()}
                    </span>
                    {getDirectionIcon(stockData.indicators.bb_lower_direction)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Historical Data Table */}
        <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
          <div className="px-6 py-4 border-b">
            <div className="flex items-center justify-between">
              <div className="flex-1"></div>
              <div className="flex items-center space-x-2">
                <h2 className="text-lg font-semibold text-gray-900">
                  Historical Data
                </h2>
                <span className="text-lg font-medium text-gray-700">
                  (
                  {new Date(
                    Date.now() - parseInt(selectedDays) * 24 * 60 * 60 * 1000
                  ).toLocaleDateString()}{" "}
                  - {new Date().toLocaleDateString()})
                </span>
              </div>
              <div className="flex-1 flex justify-end">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setSelectedDays("30")}
                    className={`px-4 py-2 rounded-lg ${
                      selectedDays === "30"
                        ? "bg-blue-100 text-blue-700 font-medium"
                        : "bg-white text-gray-600 hover:bg-gray-50"
                    }`}
                  >
                    30 Days
                  </button>
                  <button
                    onClick={() => setSelectedDays("60")}
                    className={`px-4 py-2 rounded-lg ${
                      selectedDays === "60"
                        ? "bg-blue-100 text-blue-700 font-medium"
                        : "bg-white text-gray-600 hover:bg-gray-50"
                    }`}
                  >
                    60 Days
                  </button>
                  <button
                    onClick={() => setSelectedDays("100")}
                    className={`px-4 py-2 rounded-lg ${
                      selectedDays === "100"
                        ? "bg-blue-100 text-blue-700 font-medium"
                        : "bg-white text-gray-600 hover:bg-gray-50"
                    }`}
                  >
                    100 Days
                  </button>
                </div>
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50">
                  <th className="text-left py-3 px-4 font-medium text-gray-500">
                    Date
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    Open
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    High
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    Low
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    Close
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    Volume
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    RSI
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    MA44
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    BB Upper
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500">
                    BB Lower
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {stockHistory.map((record, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="py-3 px-4">
                      {new Date(record.date).toLocaleDateString()}
                    </td>
                    <td className="text-right py-3 px-4">
                      ₹{parseFloat(record.open).toFixed(2)}
                    </td>
                    <td className="text-right py-3 px-4">
                      ₹{parseFloat(record.high).toFixed(2)}
                    </td>
                    <td className="text-right py-3 px-4">
                      ₹{parseFloat(record.low).toFixed(2)}
                    </td>
                    <td className="text-right py-3 px-4 font-medium">
                      ₹{parseFloat(record.close).toFixed(2)}
                    </td>
                    <td className="text-right py-3 px-4">
                      {parseInt(record.volume).toLocaleString()}
                    </td>
                    <td className="text-right py-3 px-4">
                      {record.rsi ? parseFloat(record.rsi).toFixed(2) : "N/A"}
                    </td>
                    <td className="text-right py-3 px-4">
                      ₹
                      {record.ma_44
                        ? parseFloat(record.ma_44).toFixed(2)
                        : "N/A"}
                    </td>
                    <td className="text-right py-3 px-4">
                      ₹
                      {record.bb_upper
                        ? parseFloat(record.bb_upper).toFixed(2)
                        : "N/A"}
                    </td>
                    <td className="text-right py-3 px-4">
                      ₹
                      {record.bb_lower
                        ? parseFloat(record.bb_lower).toFixed(2)
                        : "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockDetail;
