import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import io
import base64
from flask import Flask, request, render_template_string

class ClimateAnomalyAnalyzer:
    def __init__(self, mu, sigma, X):
        self.mu = mu
        self.sigma = sigma
        self.X = X

    def compute_zscore(self):
        return (self.X - self.mu) / self.sigma

    def compute_probabilities(self):
        Z = self.compute_zscore()
        p_below = norm.cdf(Z)
        p_above = 1 - p_below
        return p_below, p_above

    def plot_distribution(self):
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 1000)
        y = norm.pdf(x, self.mu, self.sigma)

        plt.figure(figsize=(6,4))
        plt.plot(x, y, label="Normal Distribution", linewidth=2)
        plt.axvline(self.X, color="red", linestyle="--", label=f"X = {self.X}")

        x_fill = np.linspace(self.X, self.mu + 4*self.sigma, 500)
        y_fill = norm.pdf(x_fill, self.mu, self.sigma)
        plt.fill_between(x_fill, y_fill, alpha=0.5)

        plt.title("Probability of Temperature Anomaly > X")
        plt.xlabel("Temperature Anomaly (°C)")
        plt.ylabel("Density")
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode("utf-8")

app = Flask(__name__)

@app.route("/analyze")
def analyze():
    mu = float(request.args.get("mu", 0.5))
    sigma = float(request.args.get("sigma", 0.2))
    X = float(request.args.get("X", 0.9))

    analyzer = ClimateAnomalyAnalyzer(mu, sigma, X)
    zscore = analyzer.compute_zscore()
    p_below, p_above = analyzer.compute_probabilities()
    plot_base64 = analyzer.plot_distribution()

    html = f"""
    <h2>🌡️ Climate Anomaly Analysis</h2>
    <p><b>Mean (μ):</b> {mu}</p>
    <p><b>Std Dev (σ):</b> {sigma}</p>
    <p><b>Observed Value (X):</b> {X}</p>
    <p><b>Z-score:</b> {zscore:.2f}</p>
    <p>P(X ≤ {X}): {p_below:.4f} ({p_below*100:.2f}%)</p>
    <p>P(X > {X}): {p_above:.4f} ({p_above*100:.2f}%)</p>
    <img src="data:image/png;base64,{plot_base64}" />
    """
    return render_template_string(html)

if __name__ == "__main__":
    app.run(debug=True)
