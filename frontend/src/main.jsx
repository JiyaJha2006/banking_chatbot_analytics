import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { BarChart3, KeyRound, LogIn, LogOut, MessageSquare, Mic, PanelLeftClose, PanelLeftOpen, Plus, Search, SendHorizonal, ShieldCheck, User, Volume2 } from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "/api/chat";
const CHATS_KEY = "banking-chat-chats";
const ACTIVE_CHAT_KEY = "banking-chat-active";
const SIDEBAR_KEY = "banking-chat-sidebar-collapsed";
const AUTH_KEY = "banking-chat-auth";
const TOKEN_KEY = "banking-chat-token";
const TOKEN_EXPIRES_KEY = "banking-chat-token-expires-at";
const PAGE_ROUTES = {
  login: "/login",
  signup: "/signup",
  chat: "/chat",
  newChat: "/chat/new",
  profile: "/profile",
  analytics: "/analytics",
  logout: "/logout"
};

function getCurrentPath() {
  return window.location.pathname || "/";
}

function getLanguagePath(language) {
  return `${getCurrentPath()}?language=${encodeURIComponent(language)}`;
}

function navigateTo(path) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function navigateLanguage(language) {
  navigateTo(getLanguagePath(language));
}

const copy = {
  English: {
    title: "Banking Chatbot",
    language: "Language",
    introTitle: "How can I help with banking today?",
    introCopy: "Ask about accounts, documents, fees, loans, digital banking, eligibility, or which account may suit you.",
    placeholder: "Ask a banking question",
    voice: "Voice",
    stop: "Stop",
    thinking: "Thinking...",
    newChat: "New chat",
    search: "Search chats",
    collapse: "Collapse sidebar",
    expand: "Expand sidebar",
    logout: "Log out",
    noChats: "No matching chats",
    footer: "Banking answers can vary by institution. Verify important details with your bank.",
    error: "Sorry, I could not get an answer from the backend.",
    readAnswer: "Read answer",
    relatedQuestions: "Related questions"
  },
  Hindi: {
    title: "बैंकिंग चैटबॉट",
    language: "भाषा",
    introTitle: "आज मैं बैंकिंग में आपकी कैसे मदद कर सकता हूं?",
    introCopy: "खातों, दस्तावेजों, शुल्क, लोन, डिजिटल बैंकिंग, पात्रता या सही खाते के बारे में पूछें।",
    placeholder: "बैंकिंग से जुड़ा सवाल पूछें",
    voice: "आवाज",
    stop: "रोकें",
    thinking: "सोच रहा हूं...",
    newChat: "नई चैट",
    search: "चैट खोजें",
    noChats: "कोई चैट नहीं मिली",
    footer: "बैंकिंग जानकारी संस्था के अनुसार बदल सकती है। जरूरी जानकारी अपने बैंक से जांचें।",
    error: "माफ कीजिए, backend से जवाब नहीं मिल पाया।"
  }
};

function getSpeechRecognition() {
  return window.SpeechRecognition || window.webkitSpeechRecognition;
}

function createChat() {
  const id = crypto.randomUUID();
  return { id, title: "New banking chat", messages: [], backendSessionId: "", updatedAt: Date.now() };
}

function getStoredAuthUser() {
  try {
    return JSON.parse(localStorage.getItem(AUTH_KEY) || "null");
  } catch {
    return null;
  }
}

function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function getStoredTokenExpiresAt() {
  return localStorage.getItem(TOKEN_EXPIRES_KEY) || "";
}

function isStoredTokenExpired() {
  const expiresAt = getStoredTokenExpiresAt();
  if (!expiresAt) return false;
  const expiresAtMs = Date.parse(expiresAt);
  return Number.isFinite(expiresAtMs) && expiresAtMs <= Date.now();
}

function getAuthHeaders() {
  const token = getStoredToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function getUserStorageId(user) {
  if (!user?.id || !user?.username) return "";
  const safeUsername = String(user.username).trim().toLowerCase().replace(/[^a-z0-9_-]/g, "-");
  return `user-${user.id}-${safeUsername}`;
}

function getUserChatsKey(user) {
  const userId = getUserStorageId(user);
  return userId ? `${CHATS_KEY}-${userId}` : CHATS_KEY;
}

function getUserActiveChatKey(user) {
  const userId = getUserStorageId(user);
  return userId ? `${ACTIVE_CHAT_KEY}-${userId}` : ACTIVE_CHAT_KEY;
}

function loadChatsForUser(user) {
  try {
    const stored = JSON.parse(localStorage.getItem(getUserChatsKey(user)) || "[]");
    return stored.length ? stored : [createChat()];
  } catch {
    return [createChat()];
  }
}

function loadActiveChatIdForUser(user) {
  return localStorage.getItem(getUserActiveChatKey(user)) || "";
}

function getMessageText(message) {
  return message?.message || "";
}

const authCopy = {
  English: {
    signIn: "Sign in",
    signUp: "Sign up",
    signinTitle: "Secure banking login",
    signupTitle: "Create your account",
    signinCopy: "Sign in to continue to your banking assistant.",
    signupCopy: "Sign up once and your account will be saved in MySQL.",
    username: "Username",
    password: "Password",
    language: "Language",
    submitSignin: "Sign in",
    submitSignup: "Create account",
    switchToSignup: "New user? Sign up",
    switchToSignin: "Already registered? Sign in",
    missing: "Enter both username and password.",
    genericError: "Could not complete this request."
  },
  Hindi: {
    signIn: "साइन इन",
    signUp: "साइन अप",
    signinTitle: "सुरक्षित बैंकिंग लॉगिन",
    signupTitle: "अपना अकाउंट बनाएं",
    signinCopy: "बैंकिंग असिस्टेंट जारी रखने के लिए साइन इन करें।",
    signupCopy: "एक बार साइन अप करें, आपका अकाउंट MySQL में सेव होगा।",
    username: "यूजरनेम",
    password: "पासवर्ड",
    language: "भाषा",
    submitSignin: "साइन इन",
    submitSignup: "अकाउंट बनाएं",
    switchToSignup: "नए यूजर हैं? साइन अप करें",
    switchToSignin: "पहले से रजिस्टर हैं? साइन इन करें",
    missing: "यूजरनेम और पासवर्ड दोनों भरें।",
    genericError: "यह अनुरोध पूरा नहीं हो पाया।"
  }
};

const profileCopy = {
  English: {
    profile: "Profile",
    analytics: "Analytics",
    chat: "Chat",
    account: "Account",
    userId: "User ID",
    username: "Username",
    created: "Created",
    lastSignin: "Last sign in",
    totalQuestions: "Questions asked",
    recentQuestions: "Recent questions",
    noQuestions: "No questions stored yet.",
    loading: "Loading profile...",
    error: "Could not load profile."
  },
  Hindi: {
    profile: "\u092a\u094d\u0930\u094b\u092b\u093e\u0907\u0932",
    analytics: "\u090f\u0928\u093e\u0932\u093f\u091f\u093f\u0915\u094d\u0938",
    chat: "\u091a\u0948\u091f",
    account: "\u0905\u0915\u093e\u0909\u0902\u091f",
    userId: "\u092f\u0942\u091c\u0930 ID",
    username: "\u092f\u0942\u091c\u0930\u0928\u0947\u092e",
    created: "\u092c\u0928\u093e\u092f\u093e \u0917\u092f\u093e",
    lastSignin: "\u0906\u0916\u093f\u0930\u0940 \u0938\u093e\u0907\u0928 \u0907\u0928",
    totalQuestions: "\u092a\u0942\u091b\u0947 \u0917\u090f \u0938\u0935\u093e\u0932",
    recentQuestions: "\u0939\u093e\u0932 \u0915\u0947 \u0938\u0935\u093e\u0932",
    noQuestions: "\u0905\u092d\u0940 \u0915\u094b\u0908 \u0938\u0935\u093e\u0932 \u0938\u0947\u0935 \u0928\u0939\u0940\u0902 \u0939\u0948\u0964",
    loading: "\u092a\u094d\u0930\u094b\u092b\u093e\u0907\u0932 \u0932\u094b\u0921 \u0939\u094b \u0930\u0939\u093e \u0939\u0948...",
    error: "\u092a\u094d\u0930\u094b\u092b\u093e\u0907\u0932 \u0932\u094b\u0921 \u0928\u0939\u0940\u0902 \u0939\u094b \u092a\u093e\u092f\u093e\u0964"
  }
};

function ProfilePage({ authUser, language, onAuthExpired }) {
  const [profile, setProfile] = useState(null);
  const [error, setError] = useState("");
  const p = profileCopy[language] || profileCopy.English;

  useEffect(() => {
    let isCurrent = true;
    setError("");
    fetch(`/api/profile?user_id=${encodeURIComponent(authUser?.id || "")}`, {
      headers: getAuthHeaders()
    })
      .then(async (response) => {
        const data = await response.json();
        if (response.status === 401) {
          onAuthExpired?.();
          return null;
        }
        if (!response.ok) throw new Error(data.error || p.error);
        return data.profile;
      })
      .then((data) => {
        if (isCurrent && data) setProfile(data);
      })
      .catch(() => {
        if (isCurrent) setError(p.error);
      });
    return () => {
      isCurrent = false;
    };
  }, [authUser?.id, p.error]);

  if (error) return <section className="profile-page"><p className="profile-error">{error}</p></section>;
  if (!profile) return <section className="profile-page"><p className="profile-loading">{p.loading}</p></section>;

  return (
    <section className="profile-page">
      <div className="profile-header">
        <div className="profile-avatar"><User size={28} /></div>
        <div>
          <h2>{p.account}</h2>
          <p>{profile.username}</p>
        </div>
      </div>

      <div className="profile-grid">
        <div><span>{p.userId}</span><strong>{profile.id}</strong></div>
        <div><span>{p.username}</span><strong>{profile.username}</strong></div>
        <div><span>{p.created}</span><strong>{profile.created_at || "-"}</strong></div>
        <div><span>{p.lastSignin}</span><strong>{profile.last_signin_at || "-"}</strong></div>
        <div><span>{p.totalQuestions}</span><strong>{profile.total_questions || 0}</strong></div>
      </div>

      <div className="recent-questions">
        <h3>{p.recentQuestions}</h3>
        {profile.recent_questions?.length ? (
          profile.recent_questions.map((item, index) => (
            <article key={`${item.asked_at}-${index}`}>
              <p>{item.question}</p>
              <time>{item.language} · {item.asked_at}</time>
            </article>
          ))
        ) : (
          <p className="profile-empty">{p.noQuestions}</p>
        )}
      </div>
    </section>
  );
}

const analyticsCopy = {
  English: {
    title: "FAQ Analytics Dashboard",
    subtitle: "Banking Chatbot + Analytics System",
    mostAsked: "Most asked question",
    mostTopic: "Most searched topic",
    users: "Number of users",
    averageTime: "Average response time",
    asks: "asks",
    searches: "searches",
    totalQuestions: "total questions",
    noQuestions: "No questions yet",
    noTopics: "No topics yet",
    noAnalytics: "No analytics yet.",
    loading: "Loading analytics...",
    error: "Could not load analytics.",
    topQuestions: "Top Questions",
    topTopics: "Top Topics",
    dailyQuestions: "Daily Questions"
  },
  Hindi: {
    title: "\u090f\u092b\u090f\u0915\u094d\u092f\u0942 \u090f\u0928\u093e\u0932\u093f\u091f\u093f\u0915\u094d\u0938 \u0921\u0948\u0936\u092c\u094b\u0930\u094d\u0921",
    subtitle: "\u092c\u0948\u0902\u0915\u093f\u0902\u0917 \u091a\u0948\u091f\u092c\u0949\u091f + \u090f\u0928\u093e\u0932\u093f\u091f\u093f\u0915\u094d\u0938 \u0938\u093f\u0938\u094d\u091f\u092e",
    mostAsked: "\u0938\u092c\u0938\u0947 \u091c\u094d\u092f\u093e\u0926\u093e \u092a\u0942\u091b\u093e \u0917\u092f\u093e \u0938\u0935\u093e\u0932",
    mostTopic: "\u0938\u092c\u0938\u0947 \u091c\u094d\u092f\u093e\u0926\u093e \u0916\u094b\u091c\u093e \u0917\u092f\u093e \u0935\u093f\u0937\u092f",
    users: "\u092f\u0942\u091c\u0930\u094d\u0938 \u0915\u0940 \u0938\u0902\u0916\u094d\u092f\u093e",
    averageTime: "\u0914\u0938\u0924 \u0930\u093f\u0938\u094d\u092a\u0949\u0928\u094d\u0938 \u091f\u093e\u0907\u092e",
    asks: "\u092c\u093e\u0930 \u092a\u0942\u091b\u093e",
    searches: "\u0938\u0930\u094d\u091a",
    totalQuestions: "\u0915\u0941\u0932 \u0938\u0935\u093e\u0932",
    noQuestions: "\u0905\u092d\u0940 \u0915\u094b\u0908 \u0938\u0935\u093e\u0932 \u0928\u0939\u0940\u0902",
    noTopics: "\u0905\u092d\u0940 \u0915\u094b\u0908 \u0935\u093f\u0937\u092f \u0928\u0939\u0940\u0902",
    noAnalytics: "\u0905\u092d\u0940 \u0915\u094b\u0908 \u090f\u0928\u093e\u0932\u093f\u091f\u093f\u0915\u094d\u0938 \u0928\u0939\u0940\u0902\u0964",
    loading: "\u090f\u0928\u093e\u0932\u093f\u091f\u093f\u0915\u094d\u0938 \u0932\u094b\u0921 \u0939\u094b \u0930\u0939\u093e \u0939\u0948...",
    error: "\u090f\u0928\u093e\u0932\u093f\u091f\u093f\u0915\u094d\u0938 \u0932\u094b\u0921 \u0928\u0939\u0940\u0902 \u0939\u094b \u092a\u093e\u092f\u093e\u0964",
    topQuestions: "\u091f\u0949\u092a \u0938\u0935\u093e\u0932",
    topTopics: "\u091f\u0949\u092a \u0935\u093f\u0937\u092f",
    dailyQuestions: "\u0930\u094b\u091c\u093c\u093e\u0928\u093e \u0938\u0935\u093e\u0932"
  }
};

function AnalyticsBarChart({ title, rows = [], emptyText = "No analytics yet." }) {
  const maxValue = Math.max(1, ...rows.map((row) => Number(row.value) || 0));
  return (
    <section className="analytics-chart">
      <h3>{title}</h3>
      {rows.length ? rows.map((row) => (
        <div className="chart-row" key={`${title}-${row.label}`}>
          <span title={row.label}>{row.label}</span>
          <div className="chart-track">
            <div style={{ width: `${Math.max(8, ((Number(row.value) || 0) / maxValue) * 100)}%` }}></div>
          </div>
          <strong>{row.value}</strong>
        </div>
      )) : <p className="profile-empty">{emptyText}</p>}
    </section>
  );
}

function AnalyticsPage({ language, onAuthExpired }) {
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState("");
  const a = analyticsCopy[language] || analyticsCopy.English;

  useEffect(() => {
    let isCurrent = true;
    setError("");
    fetch("/api/analytics", { headers: getAuthHeaders() })
      .then(async (response) => {
        const data = await response.json();
        if (response.status === 401) {
          onAuthExpired?.();
          return null;
        }
        if (!response.ok) throw new Error(data.error || a.error);
        return data.analytics;
      })
      .then((data) => {
        if (isCurrent && data) setAnalytics(data);
      })
      .catch(() => {
        if (isCurrent) setError(a.error);
      });
    return () => {
      isCurrent = false;
    };
  }, [a.error]);

  if (error) return <section className="profile-page"><p className="profile-error">{error}</p></section>;
  if (!analytics) return <section className="profile-page"><p className="profile-loading">{a.loading}</p></section>;

  const averageSeconds = ((Number(analytics.average_response_time_ms) || 0) / 1000).toFixed(2);

  return (
    <section className="analytics-page">
      <div className="profile-header">
        <div className="profile-avatar"><BarChart3 size={28} /></div>
        <div>
          <h2>{a.title}</h2>
          <p>{a.subtitle}</p>
        </div>
      </div>

      <div className="analytics-stats">
        <article>
          <span>{a.mostAsked}</span>
          <strong>{analytics.most_asked_question?.question || a.noQuestions}</strong>
          <small>{analytics.most_asked_question?.count || 0} {a.asks}</small>
        </article>
        <article>
          <span>{a.mostTopic}</span>
          <strong>{analytics.most_searched_topic?.topic || a.noTopics}</strong>
          <small>{analytics.most_searched_topic?.count || 0} {a.searches}</small>
        </article>
        <article>
          <span>{a.users}</span>
          <strong>{analytics.number_of_users || 0}</strong>
          <small>{analytics.total_questions || 0} {a.totalQuestions}</small>
        </article>
        <article>
          <span>{a.averageTime}</span>
          <strong>{averageSeconds}s</strong>
          <small>{analytics.average_response_time_ms || 0} ms</small>
        </article>
      </div>

      <div className="analytics-charts">
        <AnalyticsBarChart title={a.topQuestions} rows={analytics.charts?.top_questions || []} emptyText={a.noAnalytics} />
        <AnalyticsBarChart title={a.topTopics} rows={analytics.charts?.top_topics || []} emptyText={a.noAnalytics} />
        <AnalyticsBarChart title={a.dailyQuestions} rows={analytics.charts?.daily_questions || []} emptyText={a.noAnalytics} />
      </div>
    </section>
  );
}

function LoginPage({ language, setLanguage, onLogin, routePath }) {
  const [mode, setMode] = useState(routePath === PAGE_ROUTES.signup ? "signup" : "signin");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const t = authCopy[language] || authCopy.English;
  const isSignup = mode === "signup";

  useEffect(() => {
    setMode(routePath === PAGE_ROUTES.signup ? "signup" : "signin");
  }, [routePath]);

  async function submitLogin(event) {
    event.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError(t.missing);
      return;
    }
    setError("");
    try {
      const response = await fetch(isSignup ? "/api/register" : "/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || t.genericError);
      onLogin(data.user);
    } catch (err) {
      setError(err.message || t.genericError);
    }
  }

  return (
    <main className="login-page">
      <form className="login-panel" onSubmit={submitLogin}>
        <div className="login-brand">
          <div className="login-mark"><ShieldCheck size={28} /></div>
          <div>
            <h1>{isSignup ? t.signupTitle : t.signinTitle}</h1>
            <p>{isSignup ? t.signupCopy : t.signinCopy}</p>
          </div>
        </div>

        <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
          <a href={PAGE_ROUTES.login} className={!isSignup ? "active" : ""} onClick={(event) => { event.preventDefault(); setError(""); navigateTo(PAGE_ROUTES.login); }}>{t.signIn}</a>
          <a href={PAGE_ROUTES.signup} className={isSignup ? "active" : ""} onClick={(event) => { event.preventDefault(); setError(""); navigateTo(PAGE_ROUTES.signup); }}>{t.signUp}</a>
        </div>

        <div className="login-fields">
          <label>
            <span>{t.username}</span>
            <div className="login-input">
              <User size={18} />
              <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
            </div>
          </label>
          <label>
            <span>{t.password}</span>
            <div className="login-input">
              <KeyRound size={18} />
              <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={isSignup ? "new-password" : "current-password"} />
            </div>
          </label>
        </div>

        {error && <p className="login-error">{error}</p>}

        <button type="submit" className="login-submit">
          {isSignup ? <User size={18} /> : <LogIn size={18} />}
          <span>{isSignup ? t.submitSignup : t.submitSignin}</span>
        </button>

        <a
          href={isSignup ? PAGE_ROUTES.login : PAGE_ROUTES.signup}
          className="auth-switch"
          onClick={(event) => { event.preventDefault(); setError(""); navigateTo(isSignup ? PAGE_ROUTES.login : PAGE_ROUTES.signup); }}
        >
          {isSignup ? t.switchToSignin : t.switchToSignup}
        </a>

        <div className="login-language" aria-label={t.language}>
          <span>{t.language}</span>
          <div className="segment">
            {["English", "Hindi"].map((item) => (
              <a key={item} href={getLanguagePath(item)} className={language === item ? "active" : ""} onClick={(event) => { event.preventDefault(); setLanguage(item); navigateLanguage(item); }}>{item}</a>
            ))}
          </div>
        </div>
      </form>
    </main>
  );
}

function App() {
  const [language, setLanguage] = useState("English");
  const [routePath, setRoutePath] = useState(getCurrentPath());
  const [authUser, setAuthUser] = useState(getStoredAuthUser);
  const [chats, setChats] = useState(() => loadChatsForUser(getStoredAuthUser()));
  const [activeChatId, setActiveChatId] = useState(() => loadActiveChatIdForUser(getStoredAuthUser()));
  const [input, setInput] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(() => localStorage.getItem(SIDEBAR_KEY) === "true");
  const [activeView, setActiveView] = useState("chat");
  const [isThinking, setIsThinking] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const scrollRef = useRef(null);
  const searchInputRef = useRef(null);
  const recognitionRef = useRef(null);
  const typingTimerRef = useRef(null);
  const handledNewChatRouteRef = useRef("");
  const t = copy[language];
  const p = profileCopy[language] || profileCopy.English;
  const logoutLabel = t.logout || (language === "Hindi" ? "\u0932\u0949\u0917 \u0906\u0909\u091f" : "Log out");
  const canUseVoice = useMemo(() => Boolean(getSpeechRecognition()), []);

  const activeChat = chats.find((chat) => chat.id === activeChatId) || chats[0] || createChat();
  const messages = activeChat.messages || [];

  const filteredChats = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    if (!query) return chats;
    return chats.filter((chat) => {
      const haystack = [chat.title, ...(chat.messages || []).map(getMessageText)].join(" ").toLowerCase();
      return haystack.includes(query);
    });
  }, [chats, searchTerm]);

  useEffect(() => {
    if (!chats.some((chat) => chat.id === activeChatId)) {
      setActiveChatId(chats[0]?.id || "");
    }
  }, [activeChatId, chats]);

  useEffect(() => {
    if (!authUser) return;
    localStorage.setItem(getUserChatsKey(authUser), JSON.stringify(chats));
  }, [authUser, chats]);

  useEffect(() => {
    if (!authUser || !activeChatId) return;
    localStorage.setItem(getUserActiveChatKey(authUser), activeChatId);
  }, [authUser, activeChatId]);

  useEffect(() => {
    localStorage.setItem(SIDEBAR_KEY, String(isSidebarCollapsed));
  }, [isSidebarCollapsed]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isThinking]);

  useEffect(() => () => clearInterval(typingTimerRef.current), []);

  useEffect(() => {
    const syncRoute = () => {
      setRoutePath(getCurrentPath());
      const nextLanguage = new URLSearchParams(window.location.search).get("language");
      if (["English", "Hindi"].includes(nextLanguage)) setLanguage(nextLanguage);
    };
    syncRoute();
    window.addEventListener("popstate", syncRoute);
    return () => window.removeEventListener("popstate", syncRoute);
  }, []);

  useEffect(() => {
    if (!authUser) return;
    const token = getStoredToken();
    if (!token || isStoredTokenExpired()) {
      handleLogout();
      navigateTo(PAGE_ROUTES.login);
      return;
    }
    fetch("/api/token", { headers: getAuthHeaders() })
      .then((response) => {
        if (!response.ok) throw new Error("Invalid token");
      })
      .catch(() => {
        handleLogout();
        navigateTo(PAGE_ROUTES.login);
      });
  }, [authUser?.id]);

  useEffect(() => {
    if (!authUser) return;
    const checkToken = () => {
      if (isStoredTokenExpired()) {
        handleLogout();
        navigateTo(PAGE_ROUTES.login);
      }
    };
    checkToken();
    const timer = window.setInterval(checkToken, 60000);
    return () => window.clearInterval(timer);
  }, [authUser?.id]);

  useEffect(() => {
    if (!authUser) return;
    if (routePath === PAGE_ROUTES.logout) {
      handleLogout();
      navigateTo(PAGE_ROUTES.login);
      return;
    }
    if (routePath === PAGE_ROUTES.profile) {
      setActiveView("profile");
      return;
    }
    if (routePath === PAGE_ROUTES.analytics) {
      setActiveView("analytics");
      return;
    }
    if (routePath === PAGE_ROUTES.newChat && handledNewChatRouteRef.current !== routePath) {
      handledNewChatRouteRef.current = routePath;
      addNewChat(false);
      return;
    }
    if (routePath.startsWith("/chat/") && routePath !== PAGE_ROUTES.newChat) {
      const routeChatId = routePath.split("/").filter(Boolean)[1];
      if (chats.some((chat) => chat.id === routeChatId)) {
        setActiveChatId(routeChatId);
        setActiveView("chat");
      }
      return;
    }
    if (routePath === PAGE_ROUTES.chat || routePath === "/") {
      setActiveView("chat");
      if (routePath === "/") {
        navigateTo(PAGE_ROUTES.chat);
      }
    }
  }, [authUser, routePath, chats]);

  function handleLogin(user) {
    const { token, token_type, expires_at, expires_in_seconds, ...storedUser } = user;
    if (token) localStorage.setItem(TOKEN_KEY, token);
    if (expires_at) localStorage.setItem(TOKEN_EXPIRES_KEY, expires_at);
    localStorage.setItem(AUTH_KEY, JSON.stringify(storedUser));
    const userChats = loadChatsForUser(storedUser);
    setAuthUser(storedUser);
    setChats(userChats);
    setActiveChatId(loadActiveChatIdForUser(storedUser) || userChats[0]?.id || "");
    setSearchTerm("");
    setInput("");
    setActiveView("chat");
    navigateTo(PAGE_ROUTES.chat);
  }

  async function handleLogout() {
    try {
      await fetch("/api/logout", {
        method: "POST",
        headers: getAuthHeaders()
      });
    } catch {
      // Local logout should still happen if the server is unreachable.
    }
    localStorage.removeItem(AUTH_KEY);
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(TOKEN_EXPIRES_KEY);
    setAuthUser(null);
    const emptyChat = createChat();
    setChats([emptyChat]);
    setActiveChatId(emptyChat.id);
    setSearchTerm("");
    setInput("");
  }

  function handleAuthExpired() {
    handleLogout();
    navigateTo(PAGE_ROUTES.login);
  }

  function updateActiveChat(updater) {
    setChats((current) => current.map((chat) => (chat.id === activeChat.id ? updater(chat) : chat)));
  }

  function addNewChat(updateRoute = true) {
    clearInterval(typingTimerRef.current);
    setIsThinking(false);
    setIsTyping(false);
    const chat = createChat();
    setChats((current) => [chat, ...current]);
    setActiveChatId(chat.id);
    setActiveView("chat");
    setInput("");
    setSearchTerm("");
    if (updateRoute) {
      handledNewChatRouteRef.current = PAGE_ROUTES.newChat;
      navigateTo(PAGE_ROUTES.newChat);
    }
  }

  function openPage(event, path) {
    event.preventDefault();
    navigateTo(path);
  }

  function openSearchFromSidebar() {
    if (isSidebarCollapsed) {
      setIsSidebarCollapsed(false);
      window.setTimeout(() => searchInputRef.current?.focus(), 200);
      return;
    }
    searchInputRef.current?.focus();
  }

  function typeBotReply(reply, time, suggestedQuestions = [], recommendationCard = null, comparisonTable = null) {
    const botId = crypto.randomUUID();
    let index = 0;
    updateActiveChat((chat) => ({
      ...chat,
      messages: [...chat.messages, { id: botId, role: "bot", message: "", time, suggestedQuestions, recommendationCard, comparisonTable }],
      updatedAt: Date.now()
    }));
    setIsTyping(true);
    clearInterval(typingTimerRef.current);
    typingTimerRef.current = setInterval(() => {
      index += 1;
      setChats((current) => current.map((chat) => {
        if (chat.id !== activeChat.id) return chat;
        return {
          ...chat,
          messages: chat.messages.map((message) => (
            message.id === botId ? { ...message, message: reply.slice(0, index), suggestedQuestions, recommendationCard, comparisonTable } : message
          )),
          updatedAt: Date.now()
        };
      }));
      if (index >= reply.length) {
        clearInterval(typingTimerRef.current);
        setIsTyping(false);
      }
    }, 18);
  }

  async function sendMessage(text = input) {
    const trimmed = text.trim();
    if (!trimmed || isThinking || isTyping) return;
    const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const nextTitle = messages.length === 0 ? trimmed.slice(0, 42) : activeChat.title;

    updateActiveChat((chat) => ({
      ...chat,
      title: nextTitle || chat.title,
      messages: [...chat.messages, { id: crypto.randomUUID(), role: "user", message: trimmed, time: now }],
      updatedAt: Date.now()
    }));
    setInput("");
    setIsThinking(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          message: trimmed,
          language,
          session_id: activeChat.backendSessionId || null,
          user_id: authUser?.id || null,
          username: authUser?.username || ""
        })
      });
      const data = await response.json();
      if (response.status === 401) {
        handleAuthExpired();
        return;
      }
      if (!response.ok) throw new Error(data.error || "Backend error");
      setChats((current) => current.map((chat) => (
        chat.id === activeChat.id ? { ...chat, backendSessionId: data.session_id || chat.backendSessionId } : chat
      )));
      setIsThinking(false);
      typeBotReply(data.reply || "", new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), data.suggested_questions || [], data.recommendation_card || null, data.comparison_table || null);
    } catch {
      setIsThinking(false);
      typeBotReply(t.error, new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    }
  }

  function speakAnswer(text) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language === "Hindi" ? "hi-IN" : "en-IN";
    window.speechSynthesis.speak(utterance);
  }

  function toggleVoice() {
    const SpeechRecognition = getSpeechRecognition();
    if (!SpeechRecognition) return;
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = language === "Hindi" ? "hi-IN" : "en-IN";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onerror = () => setIsListening(false);
    recognition.onresult = (event) => setInput(event.results?.[0]?.[0]?.transcript || "");
    recognitionRef.current = recognition;
    recognition.start();
  }

  if (!authUser || routePath === PAGE_ROUTES.login || routePath === PAGE_ROUTES.signup) {
    return <LoginPage language={language} setLanguage={setLanguage} onLogin={handleLogin} routePath={routePath} />;
  }

  return (
    <main className={`app-frame ${isSidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="sidebar-actions">
          <a
            href="/api/buttons/collapse_sidebar"
            className="collapse-sidebar"
            onClick={(event) => { event.preventDefault(); setIsSidebarCollapsed((value) => !value); }}
            aria-label={isSidebarCollapsed ? (t.expand || "Expand sidebar") : (t.collapse || "Collapse sidebar")}
            title={isSidebarCollapsed ? (t.expand || "Expand sidebar") : (t.collapse || "Collapse sidebar")}
          >
            {isSidebarCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </a>
          <a href={PAGE_ROUTES.newChat} className="new-chat" onClick={(event) => { event.preventDefault(); addNewChat(); }} aria-label={t.newChat} title={t.newChat}>
            <Plus size={17} />
            <span>{t.newChat}</span>
          </a>
        </div>
        <div className="sidebar-view-switch">
          <a
            href={PAGE_ROUTES.chat}
            className={activeView === "chat" ? "active" : ""}
            onClick={(event) => openPage(event, PAGE_ROUTES.chat)}
            title={p.chat}
            aria-label={p.chat}
          >
            <MessageSquare size={16} />
            <span>{p.chat}</span>
          </a>
          <a
            href={PAGE_ROUTES.profile}
            className={activeView === "profile" ? "active" : ""}
            onClick={(event) => openPage(event, PAGE_ROUTES.profile)}
            title={p.profile}
            aria-label={p.profile}
          >
            <User size={16} />
            <span>{p.profile}</span>
          </a>
          <a
            href={PAGE_ROUTES.analytics}
            className={activeView === "analytics" ? "active" : ""}
            onClick={(event) => openPage(event, PAGE_ROUTES.analytics)}
            title={p.analytics || "Analytics"}
            aria-label={p.analytics || "Analytics"}
          >
            <BarChart3 size={16} />
            <span>{p.analytics || "Analytics"}</span>
          </a>
        </div>
        <div className="sidebar-search">
          <label className="search-box" title={t.search} onClick={openSearchFromSidebar}>
            <Search size={16} />
            <input ref={searchInputRef} value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} placeholder={t.search} />
          </label>
        </div>
        <nav className="chat-list" aria-label={t.search}>
          {filteredChats.length === 0 && <span className="empty-search">{t.noChats}</span>}
          {filteredChats.map((chat) => (
            <a
              key={chat.id}
              href={`/chat/${chat.id}`}
              className={chat.id === activeChat.id ? "active" : ""}
              onClick={(event) => { event.preventDefault(); setActiveChatId(chat.id); setActiveView("chat"); navigateTo(`/chat/${chat.id}`); }}
              title={chat.title || "New banking chat"}
            >
              {chat.title || "New banking chat"}
            </a>
          ))}
        </nav>
      </aside>

      <div className="app-shell">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{t.title}</h1>
            <a href={PAGE_ROUTES.logout} className="logout-button" onClick={(event) => openPage(event, PAGE_ROUTES.logout)} title={logoutLabel} aria-label={logoutLabel}>
              <LogOut size={17} />
              <span>{logoutLabel}</span>
            </a>
          </div>
          <div className="language-control" aria-label={t.language}>
            <span>{t.language}</span>
            <div className="segment">
              {["English", "Hindi"].map((item) => (
                <a key={item} href={getLanguagePath(item)} className={language === item ? "active" : ""} onClick={(event) => { event.preventDefault(); setLanguage(item); navigateLanguage(item); }}>{item}</a>
              ))}
            </div>
          </div>
        </header>

        {activeView === "profile" ? (
          <ProfilePage authUser={authUser} language={language} onAuthExpired={handleAuthExpired} />
        ) : activeView === "analytics" ? (
          <AnalyticsPage language={language} onAuthExpired={handleAuthExpired} />
        ) : (
          <>
            <section ref={scrollRef} className="conversation" aria-live="polite">
              {messages.length === 0 && (
                <div className="intro">
                  <h2>{t.introTitle}</h2>
                  <p>{t.introCopy}</p>
                </div>
              )}
              {messages.map((message) => (
                <article key={message.id} className={`message-row ${message.role}`}>
                  {message.role === "bot" && <div className="avatar">AI</div>}
                  <div className="bubble">
                    <p>{message.message}</p>
                    {message.role === "bot" && message.message && (
                      <div className="bubble-actions">
                        <button type="button" className="read-answer" onClick={() => speakAnswer(message.message)} title={t.readAnswer || "Read answer"}>
                          <Volume2 size={15} />
                          <span>{t.readAnswer || "Read answer"}</span>
                        </button>
                      </div>
                    )}
                    {message.role === "bot" && message.recommendationCard && (
                      <section className="recommendation-card" aria-label={message.recommendationCard.title || "Recommended account"}>
                        <div className="recommendation-kicker">{message.recommendationCard.agent || "Smart account agent"}</div>
                        <h3>{message.recommendationCard.account}</h3>
                        <div className="recommendation-section">
                          <span>Why recommended</span>
                          <p>{message.recommendationCard.why}</p>
                        </div>
                        {message.recommendationCard.benefits?.length > 0 && (
                          <div className="recommendation-section">
                            <span>Benefits</span>
                            <ul>
                              {message.recommendationCard.benefits.map((benefit) => <li key={benefit}>{benefit}</li>)}
                            </ul>
                          </div>
                        )}
                        {message.recommendationCard.detected_profile?.length > 0 && (
                          <div className="profile-signals">
                            {message.recommendationCard.detected_profile.map((signal) => <span key={signal}>{signal}</span>)}
                          </div>
                        )}
                      </section>
                    )}
                    {message.role === "bot" && message.comparisonTable && (
                      <section className="comparison-card" aria-label={message.comparisonTable.title || "Product comparison"}>
                        <h3>{message.comparisonTable.title}</h3>
                        <div className="comparison-table-wrap">
                          <table>
                            <thead>
                              <tr>
                                {message.comparisonTable.columns.map((column) => <th key={column}>{column}</th>)}
                              </tr>
                            </thead>
                            <tbody>
                              {message.comparisonTable.rows.map((row) => (
                                <tr key={row.feature}>
                                  <td>{row.feature}</td>
                                  {message.comparisonTable.columns.slice(1).map((column) => <td key={`${row.feature}-${column}`}>{row[column]}</td>)}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </section>
                    )}
                    {message.role === "bot" && message.suggestedQuestions?.length > 0 && (
                      <div className="suggested-questions" aria-label={t.relatedQuestions || "Related questions"}>
                        <span>{t.relatedQuestions || "Related questions"}:</span>
                        {message.suggestedQuestions.map((question) => (
                          <button
                            key={question}
                            type="button"
                            className="suggestion-chip"
                            disabled={isThinking || isTyping}
                            onClick={() => sendMessage(question)}
                          >
                            {question}
                          </button>
                        ))}
                      </div>
                    )}
                    <time>{message.time}</time>
                  </div>
                  {message.role === "user" && <div className="avatar user-avatar">You</div>}
                </article>
              ))}
              {isThinking && (
                <article className="message-row bot">
                  <div className="avatar">AI</div>
                  <div className="bubble typing"><span></span><span></span><span></span><em>{t.thinking}</em></div>
                </article>
              )}
            </section>

            <footer className="composer-wrap">
              <p>{t.footer}</p>
              <form className="composer" onSubmit={(event) => { event.preventDefault(); sendMessage(); }}>
                <input value={input} onChange={(event) => setInput(event.target.value)} placeholder={t.placeholder} />
                <button type="button" className={`voice ${isListening ? "listening" : ""}`} onClick={toggleVoice} disabled={!canUseVoice}>
                  <Mic size={18} />
                  <span>{isListening ? t.stop : t.voice}</span>
                </button>
                <button type="submit" className="send" disabled={!input.trim() || isThinking || isTyping}>
                  <SendHorizonal size={20} />
                </button>
              </form>
            </footer>
          </>
        )}
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
