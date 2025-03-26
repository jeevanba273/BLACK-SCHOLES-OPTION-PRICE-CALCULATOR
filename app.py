import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

##############################################################################
# 1) PAGE CONFIG & CUSTOM CSS
##############################################################################
st.set_page_config(
    page_title="Black–Scholes Options Price and Greeks Calculator",
    page_icon="📈",  # change to another symbol as desired
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide Streamlit header/footer, tweak fonts, etc.
st.markdown("""
<style>
/* Hide default Streamlit header & footer */
header, .css-18e3th9 {
    visibility: hidden;
    height: 0;
}
footer {   
    visibility: hidden;
    height: 0;
}

/* Increase font sizes slightly */
body, [class*="css"] {
    font-size: 15px !important;
}

/* Optional: highlight text selection in pink. */
::selection {
    background: #FF69B4;
}
</style>
""", unsafe_allow_html=True)

##############################################################################
# 2) BLACK–SCHOLES PRICING FUNCTIONS
##############################################################################
def black_scholes_call(S, K, T, r, sigma):
    """Compute Black–Scholes European call option price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Compute Black–Scholes European put option price."""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

##############################################################################
# 3) GREEKS FUNCTION
##############################################################################
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Returns a dictionary of Greeks (Delta, Gamma, Theta, Vega, Rho)
    for a European call or put under Black–Scholes.
    """
    if T <= 0:
        # Degenerate case if no time left
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    nd1 = norm.pdf(d1)       # PDF at d1
    Nd1 = norm.cdf(d1)       # CDF at d1
    Nd2 = norm.cdf(d2)       # CDF at d2
    Nmd1 = norm.cdf(-d1)     # For put
    Nmd2 = norm.cdf(-d2)

    gamma = nd1 / (S*sigma*np.sqrt(T))
    vega  = S*np.sqrt(T)*nd1  # partial wrt sigma

    if option_type.lower() == "call":
        delta = Nd1
        theta = (- (S*nd1*sigma) / (2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*Nd2)
        rho   = K*T*np.exp(-r*T)*Nd2
    else:
        delta = Nd1 - 1
        theta = (- (S*nd1*sigma) / (2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*Nmd2)
        rho   = -K*T*np.exp(-r*T)*Nmd2

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega":  vega,
        "Rho":   rho
    }

##############################################################################
# 4) MAIN HEADING AND DESCRIPTION
##############################################################################
st.title("Black–Scholes Option Price and Greeks Calculator")

st.markdown("""
Welcome to our **Black–Scholes Option Pricing Application**, 
where you can compute **European call and put prices** and 
visualize them in interactive heatmaps. Additionally, you can 
examine the **Greeks**—Delta, Gamma, Theta, Vega, and Rho—to 
understand how your option reacts to changes in spot price, 
volatility, time decay, and more.
""")

##############################################################################
# 5) SIDEBAR INPUTS
##############################################################################
st.sidebar.title("Black-Scholes Option Price and Greeks Calculator")

st.sidebar.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/jeevanba273/"
st.sidebar.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: white;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`JEEVAN B A`</a>', unsafe_allow_html=True)

st.sidebar.header("Option Parameters")

S_input = st.sidebar.number_input("Current Price (S)", value=100.0, step=1.0)
K_input = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T_input = st.sidebar.number_input("Time to Maturity (T) (in Years)", value=1.0, step=0.25)
sigma_input = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01)
r_input = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, step=0.01)

st.sidebar.subheader("Heatmap Parameters")
spot_min = st.sidebar.number_input("Min Spot Price", value=80.0, step=1.0)
spot_max = st.sidebar.number_input("Max Spot Price", value=120.0, step=1.0)
vol_min  = st.sidebar.number_input("Min Volatility", value=0.10, step=0.01)
vol_max  = st.sidebar.number_input("Max Volatility", value=0.30, step=0.01)
num_spot_points = st.sidebar.slider("Spot Grid Points", 5, 30, 9)
num_vol_points  = st.sidebar.slider("Vol Grid Points", 5, 30, 9)

##############################################################################
# 6) PARAMETER TABLE
##############################################################################
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Selected Parameters")

params = {
    "Current Price": [f"{S_input:.4f}"],
    "Strike Price": [f"{K_input:.4f}"],
    "Time to Maturity (in Years)": [f"{T_input:.4f}"],
    "Volatility": [f"{sigma_input:.4f}"],
    "Risk-Free Rate": [f"{r_input:.4f}"]
}
df_params = pd.DataFrame(params)
st.table(df_params)

##############################################################################
# 7) SINGLE OPTION PRICE & GREEKS
##############################################################################
# Call / Put Prices
call_price_single = black_scholes_call(S_input, K_input, T_input, r_input, sigma_input)
put_price_single  = black_scholes_put(S_input, K_input, T_input, r_input, sigma_input)

st.subheader("Option Prices")

col_call, col_put = st.columns(2)

with col_call:
    st.markdown(
        f"""
        <div style="background-color:#a3ffa3;
                    padding:20px;
                    border-radius:10px;
                    text-align:center;
                    color:#000000;">
            <h3 style="margin:0;">Call Price</h3>
            <p style="font-size:24px; font-weight:bold; margin:0;">
                {call_price_single:.4f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_put:
    st.markdown(
        f"""
        <div style="background-color:#ffb3b3;
                    padding:20px;
                    border-radius:10px;
                    text-align:center;
                    color:#000000;">
            <h3 style="margin:0;">Put Price</h3>
            <p style="font-size:24px; font-weight:bold; margin:0;">
                {put_price_single:.4f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Compute Greeks for call & put
call_greeks = bs_greeks(S_input, K_input, T_input, r_input, sigma_input, option_type="call")
put_greeks  = bs_greeks(S_input, K_input, T_input, r_input, sigma_input, option_type="put")

# Display them side-by-side as well
st.markdown("### Option Greeks")

gcol_call, gcol_put = st.columns(2)

with gcol_call:
    st.markdown("**Call Greeks**")
    df_cg = pd.DataFrame([call_greeks])
    st.table(df_cg)

with gcol_put:
    st.markdown("**Put Greeks**")
    df_pg = pd.DataFrame([put_greeks])
    st.table(df_pg)

# Extra spacing before the heatmap
st.markdown("<br>", unsafe_allow_html=True)

##############################################################################
# 8) HEATMAPS
##############################################################################
st.subheader("Call & Put Heatmaps")

spot_values = np.linspace(spot_min, spot_max, num_spot_points)
vol_values  = np.linspace(vol_min,  vol_max,  num_vol_points)

call_values = np.zeros((num_vol_points, num_spot_points))
put_values  = np.zeros((num_vol_points, num_spot_points))

for i, vol_ in enumerate(vol_values):
    for j, s_ in enumerate(spot_values):
        call_values[i, j] = black_scholes_call(s_, K_input, T_input, r_input, vol_)
        put_values[i, j]  = black_scholes_put(s_, K_input, T_input, r_input, vol_)

# Increase DPI to reduce blurriness; set interpolation='nearest' for crisper grid
fig, (ax_call, ax_put) = plt.subplots(1, 2, figsize=(20, 9), dpi=200)

# Call heatmap
im_call = ax_call.imshow(
    call_values,
    cmap="coolwarm",
    origin="upper",
    aspect="auto",
    interpolation="none"
)
cbar_call = fig.colorbar(im_call, ax=ax_call, label="Option Price")
ax_call.set_title("CALL Heatmap")

ax_call.set_xticks(np.arange(num_spot_points))
ax_call.set_yticks(np.arange(num_vol_points))
ax_call.set_xticklabels([f"{s:.1f}" for s in spot_values])
ax_call.set_yticklabels([f"{v:.2f}" for v in vol_values[::-1]])
ax_call.set_xlabel("Spot Price")
ax_call.set_ylabel("Volatility")

for i in range(num_vol_points):
    for j in range(num_spot_points):
        ax_call.text(
            j, i,
            f"{call_values[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold"
        )

# Put heatmap
im_put = ax_put.imshow(
    put_values,
    cmap="coolwarm",
    origin="upper",
    aspect="auto",
    interpolation="none"
)
cbar_put = fig.colorbar(im_put, ax=ax_put, label="Option Price")
ax_put.set_title("PUT Heatmap")

ax_put.set_xticks(np.arange(num_spot_points))
ax_put.set_yticks(np.arange(num_vol_points))
ax_put.set_xticklabels([f"{s:.1f}" for s in spot_values])
ax_put.set_yticklabels([f"{v:.2f}" for v in vol_values[::-1]])
ax_put.set_xlabel("Spot Price")
ax_put.set_ylabel("Volatility")

for i in range(num_vol_points):
    for j in range(num_spot_points):
        ax_put.text(
            j, i,
            f"{put_values[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold"
        )

plt.tight_layout()
st.pyplot(fig)
