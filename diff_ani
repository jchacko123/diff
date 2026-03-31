import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FCC Vacancy Diffusion", layout="wide")

TOTAL_FRAMES = 500
BLOCK_SIZE = 100

# ============================================================
# Utility functions
# ============================================================

def build_lattice(nx, ny):
    positions = {}
    index_of = {}
    site_of = {}
    k = 0
    for j in range(ny):
        for i in range(nx):
            x = i + 0.5 * (j % 2)
            y = np.sqrt(3) / 2 * j
            positions[k] = np.array([x, y], dtype=float)
            index_of[(i, j)] = k
            site_of[k] = (i, j)
            k += 1

    def get_neighbors(i, j):
        if j % 2 == 0:
            candidates = [
                (i - 1, j), (i + 1, j),
                (i - 1, j - 1), (i, j - 1),
                (i - 1, j + 1), (i, j + 1)
            ]
        else:
            candidates = [
                (i - 1, j), (i + 1, j),
                (i, j - 1), (i + 1, j - 1),
                (i, j + 1), (i + 1, j + 1)
            ]
        return [(a, b) for (a, b) in candidates if 0 <= a < nx and 0 <= b < ny]

    n_sites = nx * ny
    neighbors = {}
    for j in range(ny):
        for i in range(nx):
            s = index_of[(i, j)]
            neighbors[s] = [index_of[p] for p in get_neighbors(i, j)]

    xy = np.array([positions[i] for i in range(n_sites)])
    return positions, index_of, site_of, neighbors, xy, n_sites


def temperature_to_params(T):
    # Strong visual dependence for educational effect
    if T <= 400:
        return {"move_prob": 0.18, "impurity_bias": 0.15, "box_color": "#DCEBFF"}
    elif T <= 900:
        return {"move_prob": 0.40, "impurity_bias": 0.35, "box_color": "#E8F5D0"}
    elif T <= 1500:
        return {"move_prob": 0.75, "impurity_bias": 0.70, "box_color": "#FFE1B3"}
    else:
        return {"move_prob": 0.95, "impurity_bias": 0.90, "box_color": "#FFC4C4"}


def generate_simulation(
    nx=10,
    ny=8,
    n_frames=TOTAL_FRAMES,
    n_impurities=6,
    n_vacancies=5,
    temperatures=None,
    trail_length=18,
    seed=12
):
    if temperatures is None:
        temperatures = [300, 700, 1400, 500, 1800]

    np.random.seed(seed)

    positions, index_of, site_of, neighbors, xy, n_sites = build_lattice(nx, ny)

    if n_impurities + n_vacancies >= n_sites:
        raise ValueError("Too many impurities + vacancies for the number of lattice sites.")

    site_type = np.array(["host"] * n_sites, dtype=object)
    site_id = np.full(n_sites, -1, dtype=int)

    all_sites = np.arange(n_sites)

    impurity_sites = np.random.choice(all_sites, size=n_impurities, replace=False)
    remaining = np.setdiff1d(all_sites, impurity_sites)
    vacancy_sites = np.random.choice(remaining, size=n_vacancies, replace=False)

    for imp_id, s in enumerate(impurity_sites):
        site_type[s] = "impurity"
        site_id[s] = imp_id

    for s in vacancy_sites:
        site_type[s] = "vacancy"

    impurity_trails = {imp_id: deque(maxlen=trail_length) for imp_id in range(n_impurities)}

    for s in range(n_sites):
        if site_type[s] == "impurity":
            impurity_trails[site_id[s]].append(positions[s].copy())

    history_type = []
    history_id = []
    history_trails = []
    history_temperature = []
    history_box_color = []

    def get_temperature(frame):
        block = min(frame // BLOCK_SIZE, len(temperatures) - 1)
        return temperatures[block]

    def snapshot(frame):
        T = get_temperature(frame)
        params = temperature_to_params(T)
        history_type.append(site_type.copy())
        history_id.append(site_id.copy())
        history_trails.append({
            k: np.array(list(v)) if len(v) > 0 else np.empty((0, 2))
            for k, v in impurity_trails.items()
        })
        history_temperature.append(T)
        history_box_color.append(params["box_color"])

    for frame in range(n_frames):
        T = get_temperature(frame)
        params = temperature_to_params(T)
        move_prob = params["move_prob"]
        impurity_bias = params["impurity_bias"]

        vacancy_list = list(np.where(site_type == "vacancy")[0])
        np.random.shuffle(vacancy_list)

        for v in vacancy_list:
            if site_type[v] != "vacancy":
                continue

            if np.random.rand() > move_prob:
                continue

            neigh = neighbors[v]
            movable = [s for s in neigh if site_type[s] != "vacancy"]
            if not movable:
                continue

            impurity_neighbors = [s for s in movable if site_type[s] == "impurity"]

            if impurity_neighbors and np.random.rand() < impurity_bias:
                chosen = np.random.choice(impurity_neighbors)
            else:
                chosen = np.random.choice(movable)

            moved_impurity_id = None
            if site_type[chosen] == "impurity":
                moved_impurity_id = site_id[chosen]

            site_type[v], site_type[chosen] = site_type[chosen], site_type[v]
            site_id[v], site_id[chosen] = site_id[chosen], site_id[v]

            if moved_impurity_id is not None:
                impurity_trails[moved_impurity_id].append(positions[v].copy())

        for s in range(n_sites):
            if site_type[s] == "impurity":
                imp_id = site_id[s]
                if len(impurity_trails[imp_id]) == 0 or not np.allclose(impurity_trails[imp_id][-1], positions[s]):
                    impurity_trails[imp_id].append(positions[s].copy())

        snapshot(frame)

    return {
        "positions": positions,
        "neighbors": neighbors,
        "xy": xy,
        "history_type": history_type,
        "history_id": history_id,
        "history_trails": history_trails,
        "history_temperature": history_temperature,
        "history_box_color": history_box_color,
        "n_impurities": n_impurities,
        "n_frames": n_frames,
        "nx": nx,
        "ny": ny,
        "n_sites": n_sites
    }


def make_animation(sim_data):
    xy = sim_data["xy"]
    neighbors = sim_data["neighbors"]
    positions = sim_data["positions"]
    history_type = sim_data["history_type"]
    history_id = sim_data["history_id"]
    history_trails = sim_data["history_trails"]
    history_temperature = sim_data["history_temperature"]
    history_box_color = sim_data["history_box_color"]
    n_impurities = sim_data["n_impurities"]
    n_frames = sim_data["n_frames"]

    impurity_palette = [
        "#E45756", "#F58518", "#72B7B2", "#54A24B", "#EECA3B",
        "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#4C78A8",
        "#6C5B7B", "#355C7D", "#C06C84", "#F67280", "#99B898"
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    for site, neighs in neighbors.items():
        x1, y1 = positions[site]
        for n in neighs:
            if n > site:
                x2, y2 = positions[n]
                ax.plot([x1, x2], [y1, y2], color="lightgray", lw=0.6, zorder=1)

    host_scatter = ax.scatter([], [], s=300, c="#4C78A8", edgecolors="k", linewidths=0.45, zorder=2)
    vac_scatter = ax.scatter([], [], s=170, c="white", edgecolors="black", linewidths=1.3, zorder=5)

    impurity_scatters = []
    trail_lines = []

    for imp_id in range(n_impurities):
        color = impurity_palette[imp_id % len(impurity_palette)]
        sc = ax.scatter([], [], s=390, c=color, edgecolors="k", linewidths=0.9, zorder=4)
        impurity_scatters.append(sc)
        line, = ax.plot([], [], lw=2.2, color=color, alpha=0.65, zorder=3)
        trail_lines.append(line)

    xmin, ymin = xy.min(axis=0) - 1.0
    xmax, ymax = xy.max(axis=0) + 1.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.text(
        0.5, 1.04,
        "FCC Impurity Diffusion via Vacancy Mechanism",
        transform=ax.transAxes,
        ha="center", va="bottom", fontsize=14
    )

    temp_text = ax.text(
        0.02, 0.98,
        "",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
    )

    frame_text = ax.text(
        0.98, 0.98,
        "",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
    )

    def update(frame):
        stype = history_type[frame]
        sid = history_id[frame]
        trails = history_trails[frame]
        T = history_temperature[frame]
        box_color = history_box_color[frame]

        host_idx = np.where(stype == "host")[0]
        vac_idx = np.where(stype == "vacancy")[0]

        host_scatter.set_offsets(xy[host_idx] if len(host_idx) else np.empty((0, 2)))
        vac_scatter.set_offsets(xy[vac_idx] if len(vac_idx) else np.empty((0, 2)))

        for imp_id in range(n_impurities):
            imp_idx = np.where((stype == "impurity") & (sid == imp_id))[0]
            if len(imp_idx):
                impurity_scatters[imp_id].set_offsets(xy[imp_idx])
            else:
                impurity_scatters[imp_id].set_offsets(np.empty((0, 2)))

            tr = trails[imp_id]
            if len(tr) > 1:
                trail_lines[imp_id].set_data(tr[:, 0], tr[:, 1])
            elif len(tr) == 1:
                trail_lines[imp_id].set_data(tr[0:1, 0], tr[0:1, 1])
            else:
                trail_lines[imp_id].set_data([], [])

        temp_text.set_text(f"Temperature: {T} K")
        temp_text.set_bbox(dict(boxstyle="round,pad=0.3", fc=box_color, ec="black", alpha=0.95))
        frame_text.set_text(f"Frame {frame + 1}/{n_frames}")

        artists = [host_scatter, vac_scatter, temp_text, frame_text]
        artists.extend(impurity_scatters)
        artists.extend(trail_lines)
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=60, blit=False)
    plt.close(fig)
    return anim


@st.cache_data(show_spinner=False)
def build_animation_html(nx, ny, n_frames, n_impurities, n_vacancies, temperatures, trail_length, seed):
    sim = generate_simulation(
        nx=nx,
        ny=ny,
        n_frames=n_frames,
        n_impurities=n_impurities,
        n_vacancies=n_vacancies,
        temperatures=temperatures,
        trail_length=trail_length,
        seed=seed
    )
    anim = make_animation(sim)
    return anim.to_jshtml()


# ============================================================
# App UI
# ============================================================

st.title("FCC Vacancy-Mediated Impurity Diffusion")
st.markdown(
    f"""
This app shows a simplified educational animation of **impurity diffusion via the vacancy mechanism**
in an **FCC-like packed lattice projection**.

- **Blue** = host atoms  
- **Colored atoms** = impurities  
- **White circles** = vacancies  
- **Colored lines** = impurity tracer paths  
- Total frames: **{TOTAL_FRAMES}**
- Temperature changes every **{BLOCK_SIZE} frames**
"""
)

with st.sidebar:
    st.header("Controls")

    nx = st.slider("Lattice width (nx)", 6, 20, 10, 1)
    ny = st.slider("Lattice height (ny)", 5, 16, 8, 1)

    n_impurities = st.slider("Number of impurities", 1, 20, 6, 1)
    n_vacancies = st.slider("Number of vacancies", 1, 20, 5, 1)
    trail_length = st.slider("Tracer trail length", 3, 80, 30, 1)
    seed = st.number_input("Random seed", value=12, step=1)

    st.subheader("Temperatures by frame block")
    T1 = st.number_input("Frames 1–100 (K)", value=300, step=50)
    T2 = st.number_input("Frames 101–200 (K)", value=700, step=50)
    T3 = st.number_input("Frames 201–300 (K)", value=1400, step=50)
    T4 = st.number_input("Frames 301–400 (K)", value=500, step=50)
    T5 = st.number_input("Frames 401–500 (K)", value=1800, step=50)

    use_custom = st.button("Generate Custom Animation", type="primary")

default_params = {
    "nx": 10,
    "ny": 8,
    "n_frames": TOTAL_FRAMES,
    "n_impurities": 6,
    "n_vacancies": 5,
    "temperatures": [300, 700, 1400, 500, 1800],
    "trail_length": 30,
    "seed": 12
}

st.subheader("Standard Animation")

default_html = build_animation_html(
    nx=default_params["nx"],
    ny=default_params["ny"],
    n_frames=default_params["n_frames"],
    n_impurities=default_params["n_impurities"],
    n_vacancies=default_params["n_vacancies"],
    temperatures=default_params["temperatures"],
    trail_length=default_params["trail_length"],
    seed=default_params["seed"]
)

components.html(default_html, height=720, scrolling=True)

st.markdown("---")
st.subheader("Custom Animation")

temperatures = [int(T1), int(T2), int(T3), int(T4), int(T5)]
n_sites = nx * ny

if n_impurities + n_vacancies >= n_sites:
    st.error("Impurities + vacancies must be less than the total number of lattice sites.")
else:
    if use_custom:
        with st.spinner("Generating 500-frame animation..."):
            custom_html = build_animation_html(
                nx=nx,
                ny=ny,
                n_frames=TOTAL_FRAMES,
                n_impurities=n_impurities,
                n_vacancies=n_vacancies,
                temperatures=temperatures,
                trail_length=trail_length,
                seed=int(seed)
            )
        components.html(custom_html, height=720, scrolling=True)
    else:
        st.info("Adjust the controls in the sidebar and click 'Generate Custom Animation'.")
