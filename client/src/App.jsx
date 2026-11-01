import API_BASE_URL from "./services/apiConfig";
import { useEffect, useState } from "react";
import AdminDashboard from "./pages/AdminDashboard";
import MilkCamera from "./components/MilkCamera";
import "./App.css";

import heroImage from "./assets/hero.png";

function App() {
  const [status, setStatus] = useState("Connecting...");
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error("Server error");
        return response.json();
      })
      .then((data) => setStatus(data.message || "Connected"))
      .catch(() => setStatus("Backend unavailable"));
  }, []);

  if (window.location.pathname === "/admin") {
    return <AdminDashboard />;
  }

  const scrollToScanner = () => {
    document.getElementById("scanner")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
    setMenuOpen(false);
  };

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#home" aria-label="MilkOptic AI home">
          <span className="brand-mark">M</span>
          <span className="brand-copy">
            <strong>MilkOptic <span>AI</span></strong>
            <small>SMART MILK ANALYSIS</small>
          </span>
        </a>

        <button
          className="mobile-menu-button"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle navigation"
          aria-expanded={menuOpen}
        >
          {menuOpen ? "✕" : "☰"}
        </button>

        <nav className={`main-nav ${menuOpen ? "nav-open" : ""}`}>
          <a href="#home" onClick={() => setMenuOpen(false)}>Home</a>
          <a href="#how-it-works" onClick={() => setMenuOpen(false)}>How it works</a>
          <a href="#features" onClick={() => setMenuOpen(false)}>Features</a>
          <button className="nav-cta" onClick={scrollToScanner}>
            Try milk scan <span>↗</span>
          </button>
        </nav>
      </header>

      <main>
        <section className="hero-section" id="home">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="eyebrow-dot" />
              A smarter way to explore milk quality
            </div>

            <h1>
              A little more clarity.
              <br />
              <span>In every glass.</span>
            </h1>

            <p className="hero-description">
              Start with a simple photo. MilkOptic AI helps you review
              your scan and access available analysis information in one place.
            </p>

            <div className="hero-actions">
              <button className="primary-button" onClick={scrollToScanner}>
                <span>◎</span> Scan your milk <span className="button-arrow">→</span>
              </button>
              <a className="secondary-link" href="#how-it-works">
                Discover how it works <span>↓</span>
              </a>
            </div>

            <div className="hero-trust">
              <span className="trust-icon">✳</span>
              <p>
                <strong>Simple for you.</strong>
                <br />
                The analysis process happens behind the scenes.
              </p>
            </div>
          </div>

          <div className="hero-visual">
            <div className="hero-image-frame">
              <img
                src={heroImage}
                alt="MilkOptic AI milk analysis"
                className="hero-image"
              />
              <div className="image-overlay" />
              <div className="floating-label label-top">
                <span className="label-icon">✳</span>
                <span>
                  <strong>MilkOptic AI</strong>
                  <small>Visual analysis platform</small>
                </span>
              </div>
              <div className="floating-label label-bottom">
                <span className="status-pulse" />
                <span>
                  <strong>Scan. Review. Understand.</strong>
                  <small>One simple experience</small>
                </span>
              </div>
            </div>
            <div className="visual-decoration decoration-one" />
            <div className="visual-decoration decoration-two" />
          </div>
        </section>

        <section className="value-strip" aria-label="Platform highlights">
          <div>
            <span className="value-icon">01</span>
            <span><strong>Easy to start</strong><small>Use your phone camera</small></span>
          </div>
          <div>
            <span className="value-icon">02</span>
            <span><strong>One place for scans</strong><small>Review scan information</small></span>
          </div>
          <div>
            <span className="value-icon">03</span>
            <span><strong>Transparent results</strong><small>Know what is measured</small></span>
          </div>
        </section>

        <section className="scanner-section" id="scanner">
          <div className="section-heading">
            <span className="section-kicker">YOUR FIRST STEP</span>
            <h2>Let's take a closer look.</h2>
            <p>
              Capture or upload a photo to begin. Follow the on-screen
              instructions for the best image.
            </p>
          </div>

          <div className="scanner-status">
            <span className={`status-dot ${status === "Backend unavailable" ? "status-offline" : ""}`} />
            <span>Service status: {status}</span>
          </div>

          <div className="scanner-panel">
            <MilkCamera />
          </div>

          <p className="science-note">
            <strong>Important:</strong> A normal milk photo cannot by itself
            confirm chemical purity, identify every adulterant, or establish
            health safety. Only validated tests can confirm the measurements
            they are designed to assess.
          </p>
        </section>

        <section className="how-section" id="how-it-works">
          <div className="section-heading">
            <span className="section-kicker">HOW IT WORKS</span>
            <h2>Simple on the surface. Thoughtful underneath.</h2>
            <p>Three straightforward steps to begin your scan.</p>
          </div>

          <div className="steps-grid">
            <article className="step-card">
              <span className="step-number">01</span>
              <div className="step-icon">▧</div>
              <h3>Capture a photo</h3>
              <p>Use your phone camera or upload an existing image of your sample.</p>
            </article>
            <article className="step-card">
              <span className="step-number">02</span>
              <div className="step-icon">⌘</div>
              <h3>Review the analysis</h3>
              <p>See the available image checks and the status of your scan.</p>
            </article>
            <article className="step-card">
              <span className="step-number">03</span>
              <div className="step-icon">▤</div>
              <h3>Understand your report</h3>
              <p>Review reported measurements and whether further testing is needed.</p>
            </article>
          </div>
        </section>

        <section className="features-section" id="features">
          <div className="features-copy">
            <span className="section-kicker">BUILT AROUND CLARITY</span>
            <h2>Technology you can understand.</h2>
            <p>
              MilkOptic AI is being developed to make sample analysis easier
              to access, follow and understand—without hiding the limitations
              behind a result.
            </p>
          </div>

          <div className="feature-list">
            <article>
              <span>✓</span>
              <div>
                <h3>Simple scan experience</h3>
                <p>Camera capture and image upload in one workflow.</p>
              </div>
            </article>
            <article>
              <span>✓</span>
              <div>
                <h3>Scan status visibility</h3>
                <p>Understand whether a scan is processing or completed.</p>
              </div>
            </article>
            <article>
              <span>✓</span>
              <div>
                <h3>Honest reporting</h3>
                <p>Distinguish image-based observations from confirmed laboratory measurements.</p>
              </div>
            </article>
          </div>
        </section>

        <section className="closing-cta">
          <div>
            <span className="section-kicker">READY WHEN YOU ARE</span>
            <h2>Your next scan starts here.</h2>
            <p>Bring your sample photo. We'll guide you through the next step.</p>
          </div>
          <button className="primary-button" onClick={scrollToScanner}>
            Start a scan <span className="button-arrow">→</span>
          </button>
        </section>
      </main>

      <footer className="site-footer">
        <a className="brand footer-brand" href="#home">
          <span className="brand-mark">M</span>
          <span className="brand-copy">
            <strong>MilkOptic <span>AI</span></strong>
            <small>SMART MILK ANALYSIS</small>
          </span>
        </a>
        <p>Built to make milk analysis easier to understand.</p>
        <span className="footer-copy">© {new Date().getFullYear()} MilkOptic AI</span>
      </footer>
    </div>
  );
}

export default App;
