import API_BASE_URL from "./apiConfig";
const API_BASE = `${API_BASE_URL}/api`;;

export async function adminLogin(username: string, password: string) {
  const response = await fetch(`${API_BASE}/admin/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.message || "Login failed");
  }

  return data as { success: boolean; token: string; expiresIn: number };
}

export async function getAdminScans(token: string) {
  const response = await fetch(`${API_BASE}/admin/scans`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.message || "Could not load scans");
  }

  return data as {
    success: boolean;
    total: number;
    scans: {
      _id: string;
      status: string;
      analysisType: string;
      createdAt: string;
    }[];
  };
}
