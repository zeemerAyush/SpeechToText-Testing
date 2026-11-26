import {
  ApiError,
  clearSession,
  errorMessage,
  listRecordings,
  loadSession,
  login,
  saveSession,
  setUnauthorizedHandler,
} from "./api";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
  setUnauthorizedHandler(null);
});

describe("errorMessage", () => {
  it("reads FastAPI error shapes", () => {
    expect(errorMessage(400, { detail: "Bad file." })).toBe("Bad file.");
    expect(errorMessage(422, { detail: [{ msg: "x" }] })).toMatch(/invalid/i);
    expect(errorMessage(500, null)).toBe("Request failed (500).");
  });
});

describe("session storage", () => {
  it("round-trips and clears", () => {
    expect(loadSession()).toBeNull();
    saveSession({ token: "t", role: "admin", username: "a" });
    expect(loadSession()).toEqual({ token: "t", role: "admin", username: "a" });
    clearSession();
    expect(loadSession()).toBeNull();
  });
});

describe("login", () => {
  it("returns the session on success", async () => {
    mockFetch(200, { access_token: "tok", token_type: "bearer", role: "uploader" });
    expect(await login("sam", "pw")).toEqual({ token: "tok", role: "uploader", username: "sam" });
  });
  it("throws the server message on failure", async () => {
    mockFetch(401, { detail: "Invalid username or password." });
    await expect(login("sam", "bad")).rejects.toThrow("Invalid username or password.");
  });
});

describe("authenticated requests", () => {
  it("send the bearer token", async () => {
    saveSession({ token: "abc", role: "viewer", username: "v" });
    const fetchMock = mockFetch(200, []);
    await listRecordings();
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer abc");
  });
  it("on 401 clear the session and notify the app", async () => {
    saveSession({ token: "expired", role: "viewer", username: "v" });
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(401, { detail: "Authentication required." });
    await expect(listRecordings()).rejects.toBeInstanceOf(ApiError);
    expect(loadSession()).toBeNull();
    expect(handler).toHaveBeenCalledOnce();
  });
});
