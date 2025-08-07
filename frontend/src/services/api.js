import axios from "axios";

// API base URL
const API_BASE_URL = "http://localhost:8000";
const WS_BASE_URL = "ws://localhost:8000";

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// WebSocket connection manager
class WebSocketManager {
  constructor() {
    this.liveSocket = null;
    this.signalsSocket = null;
    this.onLiveDataUpdate = null;
    this.onSignalsUpdate = null;
    this.onSystemStatusUpdate = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.isConnecting = false;
    this.connectionState = "disconnected";
  }

  // Connect to live data WebSocket
  connectLiveData(onUpdate, onStatusUpdate) {
    this.onLiveDataUpdate = onUpdate;
    this.onSystemStatusUpdate = onStatusUpdate;

    if (this.isConnecting) {
      console.log("Already attempting to connect...");
      return;
    }

    this.isConnecting = true;

    try {
      this.liveSocket = new WebSocket(`${WS_BASE_URL}/ws/live`);

      this.liveSocket.onopen = () => {
        console.log("Live data WebSocket connected");
        this.reconnectAttempts = 0;
        this.isConnecting = false;
        this.connectionState = "connected";

        // Send initial ping to establish connection
        this.ping();
      };

      this.liveSocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          switch (data.type) {
            case "connection":
              console.log("Live data connection:", data.message);
              break;
            case "live_update":
              if (this.onLiveDataUpdate) {
                this.onLiveDataUpdate(data.data);
              }
              break;
            case "system_status":
              if (this.onSystemStatusUpdate) {
                this.onSystemStatusUpdate(data.data);
              }
              break;
            case "pong":
              // Handle pong response
              break;
            default:
              console.log("Unknown message type:", data.type);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      this.liveSocket.onclose = (event) => {
        console.log(
          "Live data WebSocket disconnected",
          event.code,
          event.reason
        );
        this.connectionState = "disconnected";
        this.isConnecting = false;

        // Only attempt reconnection if it wasn't a clean close
        if (event.code !== 1000) {
          this.scheduleReconnect();
        }
      };

      this.liveSocket.onerror = (error) => {
        console.error("Live data WebSocket error:", error);
        this.connectionState = "error";
        this.isConnecting = false;
      };
    } catch (error) {
      console.error("Failed to connect to live data WebSocket:", error);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  // Connect to signals WebSocket
  connectSignals(onUpdate) {
    this.onSignalsUpdate = onUpdate;

    try {
      this.signalsSocket = new WebSocket(`${WS_BASE_URL}/ws/signals`);

      this.signalsSocket.onopen = () => {
        console.log("Signals WebSocket connected");
      };

      this.signalsSocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          switch (data.type) {
            case "connection":
              console.log("Signals connection:", data.message);
              break;
            case "signal_update":
              if (this.onSignalsUpdate) {
                this.onSignalsUpdate(data.data);
              }
              break;
            case "pong":
              // Handle pong response
              break;
            default:
              console.log("Unknown message type:", data.type);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      this.signalsSocket.onclose = () => {
        console.log("Signals WebSocket disconnected");
      };

      this.signalsSocket.onerror = (error) => {
        console.error("Signals WebSocket error:", error);
      };
    } catch (error) {
      console.error("Failed to connect to signals WebSocket:", error);
    }
  }

  // Schedule reconnection
  scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

      console.log(
        `Scheduling reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`
      );

      setTimeout(() => {
        console.log(
          `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`
        );
        if (this.onLiveDataUpdate && this.onSystemStatusUpdate) {
          this.connectLiveData(
            this.onLiveDataUpdate,
            this.onSystemStatusUpdate
          );
        }
      }, delay);
    } else {
      console.error("Max reconnection attempts reached");
    }
  }

  // Send ping to keep connection alive
  ping() {
    if (this.liveSocket && this.liveSocket.readyState === WebSocket.OPEN) {
      try {
        this.liveSocket.send(JSON.stringify({ type: "ping" }));
      } catch (error) {
        console.error("Error sending ping:", error);
      }
    }
    if (
      this.signalsSocket &&
      this.signalsSocket.readyState === WebSocket.OPEN
    ) {
      try {
        this.signalsSocket.send(JSON.stringify({ type: "ping" }));
      } catch (error) {
        console.error("Error sending ping:", error);
      }
    }
  }

  // Get connection state
  getConnectionState() {
    return this.connectionState;
  }

  // Check if connected
  isConnected() {
    return this.liveSocket && this.liveSocket.readyState === WebSocket.OPEN;
  }

  // Disconnect WebSockets
  disconnect() {
    if (this.liveSocket) {
      this.liveSocket.close(1000, "User disconnect");
      this.liveSocket = null;
    }
    if (this.signalsSocket) {
      this.signalsSocket.close(1000, "User disconnect");
      this.signalsSocket = null;
    }
    this.connectionState = "disconnected";
    this.isConnecting = false;
  }
}

// API service class
class StockScreenerAPI {
  constructor() {
    this.wsManager = new WebSocketManager();
  }

  // Health check
  async getHealth() {
    try {
      const response = await api.get("/health");
      return response.data;
    } catch (error) {
      console.error("Health check failed:", error);
      throw error;
    }
  }

  // Get all stock signals (BUY/SELL/HOLD)
  async getStockSignals() {
    try {
      const response = await api.get("/api/stocks/signals");
      return response.data;
    } catch (error) {
      console.error("Failed to fetch stock signals:", error);
      throw error;
    }
  }

  // Get detailed data for a specific stock
  async getStockDetails(symbol) {
    try {
      const response = await api.get(`/api/stocks/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Failed to fetch stock details for ${symbol}:`, error);
      throw error;
    }
  }

  // Get historical data for a specific stock
  async getStockHistory(symbol, days = 30) {
    try {
      const response = await api.get(
        `/api/stocks/${symbol}/history?days=${days}`
      );
      return response.data;
    } catch (error) {
      console.error(`Failed to fetch stock history for ${symbol}:`, error);
      throw error;
    }
  }

  // Get popular stocks data
  async getPopularStocks() {
    try {
      const response = await api.get("/api/stocks/popular");
      return response.data;
    } catch (error) {
      console.error("Failed to fetch popular stocks:", error);
      throw error;
    }
  }

  // Get system status
  async getSystemStatus() {
    try {
      const response = await api.get("/health");
      return response.data;
    } catch (error) {
      console.error("Failed to fetch system status:", error);
      throw error;
    }
  }

  // WebSocket methods
  connectLiveData(onUpdate, onStatusUpdate) {
    this.wsManager.connectLiveData(onUpdate, onStatusUpdate);
  }

  connectSignals(onUpdate) {
    this.wsManager.connectSignals(onUpdate);
  }

  disconnectWebSockets() {
    this.wsManager.disconnect();
  }

  pingWebSockets() {
    this.wsManager.ping();
  }

  // Get WebSocket connection state
  getWebSocketState() {
    return this.wsManager.getConnectionState();
  }

  // Check if WebSocket is connected
  isWebSocketConnected() {
    return this.wsManager.isConnected();
  }
}

// Create and export API instance
const stockScreenerAPI = new StockScreenerAPI();

export default stockScreenerAPI;
