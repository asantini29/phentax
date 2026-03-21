device = "3"
if len(device) == 0:
    platform = "cpu"
    target_directory = "cpu"
else:
    platform = "gpu"
    target_directory = "gpu"

import os

os.makedirs(target_directory, exist_ok=True)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = device

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from lisaconstants import ASTRONOMICAL_YEAR
from tqdm import tqdm

from phentax.waveform import IMRPhenomTHM

try:
    import pysco
    prd_style = pysco.plots.journals.get_style("paper", journal="prd", cols="onecol")
    plt.style.use(prd_style)
    cmap = pysco.plots.get_cmap("seq_r")
    colors = pysco.plots.get_colorslist('cat')
    # set the color cycle to the colors list
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

except (ImportError, ModuleNotFoundError):
    print("pysco not found, using default matplotlib style and colormap")
    cmap = plt.get_cmap("cividis")

# ---------------
tlowfit = True  # use a fit to set the starting time of the root finder used in t(f)
tol = 1e-12  # root finding tolerance
Tobs = 2 * ASTRONOMICAL_YEAR / 12
dt = 10.0
# ---------------

chi1 = 0.9
chi2 = 0.3
distance = 500.0
inclination = jnp.pi / 3.0
phi_ref = 0.0
psi = 1.0
f_min = 5e-5
delta_t = 10
f_ref = f_min

Mt_min, Mt_max = 5e4, 5e7
qmin, qmax = 0.1, 1.0

num_per_axis = 20
Mt_values = jnp.logspace(jnp.log10(Mt_min), jnp.log10(Mt_max), num_per_axis)
q_values = jnp.linspace(qmin, qmax, num_per_axis)
N_AVG = 50  # number of times to repeat each computation for averaging

batch_sizes = [1, 10, 50, 100, 200]

def mt_q_to_m1_m2(mt, q):
    """
    Transform total mass and mass ratio to component masses, assuming m1 >= m2 and q = m2/m1 <= 1.
    """
    m1 = mt * 1 / (1 + q)
    m2 = mt * q / (1 + q)
    return m1, m2


def fill_batch_arrays(
    batch_size, key, m1, m2, chi1, chi2, distance, phi_ref, inclination, psi
):
    """ """
    key, subkey = jax.random.split(key)
    random_params = jax.random.uniform(subkey, (batch_size, 8))

    m1_batch = (1 + 0.7 * random_params[:, 0]) * m1
    m2_batch = (1 + 0.5 * random_params[:, 1]) * m2
    chi1_batch = (1 + 0.1 * random_params[:, 2]) * chi1
    chi2_batch = (1 + 0.1 * random_params[:, 3]) * chi2
    distance_batch = (1 + 0.1 * random_params[:, 4]) * distance
    phi_ref_batch = (1 + 0.1 * random_params[:, 5]) * phi_ref
    psi_batch = (1 + 0.1 * random_params[:, 6]) * psi
    inclination_batch = (1 + 0.1 * random_params[:, 7]) * inclination

    return (
        key,
        m1_batch,
        m2_batch,
        chi1_batch,
        chi2_batch,
        distance_batch,
        phi_ref_batch,
        inclination_batch,
        psi_batch,
    )

def plot_Mq_times(Mt_values, q_values, times_list):
    Mt_grid, q_grid = np.meshgrid(Mt_values, q_values, indexing="ij")
    times_array = np.array(times_list).reshape(len(Mt_values), len(q_values))
    # limit the time to 3 decimal places for better color scaling
    # times_array = times_array / 1e3 # convert to milliseconds
    plt.figure()
    cp = plt.contourf(Mt_grid, q_grid, times_array, levels=20, cmap=cmap)
    # format the time values in the colorbar to be in milliseconds with 2 decimal places
    cbar = plt.colorbar(cp)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
    cbar.set_label("Average wall time (ms)", rotation=90, labelpad=12)
    plt.xscale("log")
    plt.xlabel(r"Total Mass ($M_{\rm tot}$)")
    plt.ylabel(r"Mass Ratio ($q$)")
    plt.savefig(f"{target_directory}/timings_Mt_q.png")
    # plt.show()

def plot_batch_times(batch_sizes, batch_times, warmup_batch_times):
    plt.figure()
    plt.plot(batch_sizes, batch_times, marker=".", label="Batch Time")
    plt.plot(batch_sizes, warmup_batch_times, marker="x", label="Warmup Time")
    plt.plot(batch_sizes, np.array(batch_times) / np.array(batch_sizes), marker=".", label="Average time per sample")
    plt.plot(batch_sizes, batch_times[0] / batch_sizes[0] * np.array(batch_sizes), c='k', linestyle="--", label="Linear Scaling Reference")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Batch Size ($N_{\rm batch}$)")
    plt.ylabel(r"Time (seconds)")
    # plt.title("Average wall time (seconds) vs Batch Size")
    # place the legend above the plot without overlapping the data points
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fancybox=True, numpoints=1)
    #plt.grid(True)
    plt.savefig(f"{target_directory}/timings_batch_size.png", dpi=300)
    # plt.show()

if __name__ == "__main__":

    wave_gen = IMRPhenomTHM(
        higher_modes="all",
        include_negative_modes=True,  # negative m modes will be produced by simmetry
        t_low_fit=tlowfit,
        coarse_grain=False,  # if false it will generate the waveform on a dense time grid with the specified timestep
        atol=tol,
        rtol=tol,
        T=Tobs,
    )

    # warm up
    m1, m2 = mt_q_to_m1_m2(Mt_values[0], q_values[0])

    tic = time.time()
    times, mask, h_plus, h_cross = wave_gen.compute_polarizations_at_once(
        m1,
        m2,
        chi1,
        chi2,
        distance,
        phi_ref,
        f_ref,
        f_min,
        inclination,
        psi,
        delta_t=delta_t,
    )

    h_plus.block_until_ready()
    toc = time.time()
    print(f"Warmup time: {toc - tic:.2f} seconds")

    times_list = []
    for mt in tqdm(Mt_values, desc="Total Mass"):
        for q in tqdm(q_values, desc="Mass Ratio"):
            m1, m2 = mt_q_to_m1_m2(mt, q)
            elapsed_time = 0.0
            for _ in range(N_AVG):
                tic = time.time()
                times, mask, h_plus, h_cross = wave_gen.compute_polarizations_at_once(
                    m1,
                    m2,
                    chi1,
                    chi2,
                    distance,
                    phi_ref,
                    f_ref,
                    f_min,
                    inclination,
                    psi,
                    delta_t=delta_t,
                )
                h_plus.block_until_ready()
                toc = time.time()
                elapsed_time += toc - tic
            times_list.append(elapsed_time / N_AVG)
            print(
                f"Mt: {mt:.2e}, q: {q:.2f}, average time: {elapsed_time / N_AVG:.2f} seconds"
            )

    plot_Mq_times(Mt_values, q_values, times_list)

    # ------
    # Now do the same for different batch sizes, fixing Mt and q
    mt = 1e6
    q = 0.7
    m1, m2 = mt_q_to_m1_m2(mt, q)

    warmup_batch_times = []
    batch_times = []
    key = jax.random.PRNGKey(0)

    for batch_size in batch_sizes:
        (
            key,
            m1_batch,
            m2_batch,
            chi1_batch,
            chi2_batch,
            distance_batch,
            phi_ref_batch,
            inclination_batch,
            psi_batch,
        ) = fill_batch_arrays(
            batch_size, key, m1, m2, chi1, chi2, distance, phi_ref, inclination, psi
        )

        tic = time.time()
        times, mask, h_plus_batch, h_cross_batch = (
            wave_gen.compute_polarizations_at_once(
                m1_batch,
                m2_batch,
                chi1_batch,
                chi2_batch,
                distance_batch,
                phi_ref_batch,
                f_ref,
                f_min,
                inclination_batch,
                psi_batch,
                delta_t=delta_t,
            )
        )
        h_plus_batch.block_until_ready()
        toc = time.time()
        warmup_batch_times.append(toc - tic)
        print(f"Batch size: {batch_size}, warmup time: {toc - tic:.2f} seconds")

        # ---- Now do the actual timing with multiple runs for averaging
        elapsed_time = 0.0
        for _ in range(N_AVG):
            (
                key,
                m1_batch,
                m2_batch,
                chi1_batch,
                chi2_batch,
                distance_batch,
                phi_ref_batch,
                inclination_batch,
                psi_batch,
            ) = fill_batch_arrays(
                batch_size, key, m1, m2, chi1, chi2, distance, phi_ref, inclination, psi
            )

            tic = time.time()
            times, mask, h_plus_batch, h_cross_batch = (
                wave_gen.compute_polarizations_at_once(
                    m1_batch,
                    m2_batch,
                    chi1_batch,
                    chi2_batch,
                    distance_batch,
                    phi_ref_batch,
                    f_ref,
                    f_min,
                    inclination_batch,
                    psi_batch,
                    delta_t=delta_t,
                )
            )
            h_plus_batch.block_until_ready()
            toc = time.time()
            elapsed_time += toc - tic
        batch_times.append(elapsed_time / N_AVG)
        print(
            f"Batch size: {batch_size}, average time: {elapsed_time / N_AVG:.2f} seconds"
        )

    plot_batch_times(batch_sizes, batch_times, warmup_batch_times)
    print("Done!")