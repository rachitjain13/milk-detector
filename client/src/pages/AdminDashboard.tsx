
import { useCallback, useEffect, useState } from "react";
import { adminLogin, getAdminScans } from "../services/AdminApi";
import "./AdminDashboard.css";

type Scan = {
  _id: string;
  status: string;
  analysisType: string;
  createdAt: string;
};

export default function AdminDashboard() {
  const [username, setUsername] = useState("owner");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState<string | null>(
    () => sessionStorage.getItem("milkoptic_admin_token")
  );
  const [scans, setScans] = useState<Scan[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadScans = useCallback(async (currentToken: string) => {
    setLoading(true);
    setError("");

    try {
      const data = await getAdminScans(currentToken);
      setScans(data.scans);
      setTotal(data.total);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed";
      setError(message);

      if (message.includes("session") || message.includes("token")) {
        sessionStorage.removeItem("milkoptic_admin_token");
        setToken(null);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (token) void loadScans(token);
  }, [token, loadScans]);

  async function handleLogin(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const data = await adminLogin(username, password);
      sessionStorage.setItem("milkoptic_admin_token", data.token);
      setToken(data.token);
      setPassword("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  function logout() {
    sessionStorage.removeItem("milkoptic_admin_token");
    setToken(null);
    setScans([]);
    setTotal(0);
    setError("");
  }

  if (!token) {
    return (
      <main className="admin-page">
        <form className="login-card" onSubmit={handleLogin}>
          <div className="admin-logo">🥛</div>
          <p className="eyebrow">PRIVATE WORKSPACE</p>
          <h1>MilkOptic AI</h1>
          <p className="muted">Owner dashboard sign in</p>

          <label htmlFor="username">Username</label>
          <input
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
          />

          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />

          {error && <p className="error-message">{error}</p>}

          <button className="admin-primary" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>
      </main>
    );
  }

  return (
    <main className="admin-page">
      <header className="admin-header">
        <div>
          <p className="eyebrow">OWNER WORKSPACE</p>
          <h1>MilkOptic Dashboard</h1>
          <p className="muted">Incoming scan management</p>
        </div>
        <button className="admin-secondary" onClick={logout}>
          Sign out
        </button>
      </header>

      <section className="admin-stats">
        <article className="stat-card">
          <span>Total scans</span>
          <strong>{total}</strong>
        </article>
        <article className="stat-card">
          <span>Latest scans shown</span>
          <strong>{scans.length}</strong>
        </article>
        <article className="stat-card">
          <span>Research status</span>
          <strong>Experimental</strong>
        </article>
      </section>

      <section className="scan-panel">
        <div className="panel-heading">
          <div>
            <h2>Recent scans</h2>
            <p className="muted">Latest 100 records</p>
          </div>
          <button
            className="admin-secondary"
            onClick={() => void loadScans(token)}
            disabled={loading}
          >
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>

        {error && <p className="error-message">{error}</p>}

        {!loading && !error && scans.length === 0 && (
          <div className="empty-state">
            <span>📷</span>
            <h3>No scans yet</h3>
            <p>Uploaded milk photos will appear here.</p>
          </div>
        )}

        {scans.length > 0 && (
          <div className="table-wrap">
            <table className="scan-table">
              <thead>
                <tr>
                  <th>Scan ID</th>
                  <th>Date</th>
                  <th>Status</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody>
                {scans.map((scan) => (
                  <tr key={scan._id}>
                    <td className="scan-id">{scan._id}</td>
                    <td>{new Date(scan.createdAt).toLocaleString()}</td>
                    <td>
                      <span className="status-pill">{scan.status}</span>
                    </td>
                    <td>{scan.analysisType}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <p className="admin-footnote">
        Image-based research only. Scan records are not verified adulterant
        detections.
      </p>
    </main>
  );
}
