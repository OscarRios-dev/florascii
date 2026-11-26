#!/usr/bin/env python3
"""bouquet3d — un ramo de baby's breath (gypsophila) y rosas rosadas hecho de texto
("forever my favorite"). Los tallos crecen, las florecitas se abren una a una, el ramo
gira con una brisa suave y al final se deshace con el viento.

Uso:
  python bouquet3d.py                          # ventana en tiempo real (Esc sale, Espacio pausa, R reinicia)
  python bouquet3d.py --record ramo.mp4        # exporta 1 ciclo a MP4 (vertical 1080x1920 por defecto)
  python bouquet3d.py --still 9.5 ramo.png     # guarda un solo cuadro (segundo 9.5) como imagen
  python bouquet3d.py --record ramo.mp4 --size 1080x1350 --loops 2
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np

PHRASE = "forever my favorite "
CAPTION = "forever my favorite"

# --- línea de tiempo (segundos) ---------------------------------------------------
LOOP = 15.0          # duración de un ciclo completo
T_START = 0.35       # los tallos empiezan a crecer
T_GROW = 4.6         # tiempo que tardan los tallos en llegar a la punta
T_RIBBON = 2.3       # el listón se anuda
T_DIS = 11.0         # empieza la desintegración con el viento
SPIN = 0.42          # giro (rad/s)
ELEV = 0.30          # cámara un poco por encima (rad)

# --- colores (RGB), todos suaves: nada saturado ------------------------------------
WHITE = np.array([252, 249, 245], float)
BLUSH_TINT = np.array([248, 212, 224], float)
EYE = np.array([226, 214, 138], float)          # centro cremoso de la flor
BUD_GREEN = np.array([178, 204, 160], float)
SAGE = np.array([118, 148, 104], float)
SAGE_DARK = np.array([84, 112, 82], float)
RIBBON = np.array([236, 168, 186], float)
DUST = np.array([255, 216, 228], float)
ROSE_HI = np.array([255, 232, 238], float)
POLLEN = np.array([255, 236, 200], float)
ROSE_PALETTES = [  # (oscuro, medio, claro) en rosas empolvados
    (np.array([96, 40, 62], float), np.array([238, 168, 188], float)),
    (np.array([92, 32, 56], float), np.array([224, 132, 162], float)),
    (np.array([100, 48, 58], float), np.array([244, 182, 176], float)),
]

UP = np.array([0.0, 0.0, 1.0])
DS = 0.0105          # separación entre puntos de un tallo


def clamp01(v):
    return np.clip(v, 0.0, 1.0)


def ease_out_cubic(u):
    u = clamp01(u)
    return 1 - (1 - u) ** 3


def ease_in_out(u):
    u = clamp01(u)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def perp(rng, d):
    r = rng.normal(size=3)
    r = r - d * np.dot(r, d)
    return r / np.linalg.norm(r)


def rot_axis(v, k, ang):
    """Rodrigues: gira v alrededor del eje unitario k."""
    return v * math.cos(ang) + np.cross(k, v) * math.sin(ang) + k * np.dot(k, v) * (1 - math.cos(ang))


def bezier(p0, p1, p2, p3, n=400):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def resample(pts, ds):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    tt = np.arange(0, cum[-1], ds)
    return np.stack([np.interp(tt, cum, pts[:, i]) for i in range(3)], -1), tt


def rotate_view(P, angle, elev):
    """Gira alrededor del eje vertical (z) y proyecta a una cámara algo elevada.
    Devuelve (derecha, arriba, hacia_cámara)."""
    ca, sa = math.cos(angle), math.sin(angle)
    x1 = P[:, 0] * ca - P[:, 1] * sa
    y1 = P[:, 0] * sa + P[:, 1] * ca
    z1 = P[:, 2]
    ce, se = math.cos(elev), math.sin(elev)
    return x1, y1 * se + z1 * ce, -y1 * ce + z1 * se


def reach_time(u, umax):
    """Instante en que la punta del tallo llega a la longitud u (inversa de ease_in_out)."""
    s = clamp01(u / umax)
    return T_START + T_GROW * np.arccos(1 - 2 * s) / np.pi


# =====================================================================================
#  Rosa rosada (superficie paramétrica clásica) — cada rosa con su tamaño, giro y paleta
# =====================================================================================
class Rose:
    """Rosa hecha de pétalos explícitos: cada pétalo es un parche curvo; los de adentro
    forman un capullo cerrado y los de afuera se abren y caen hacia atrás al florecer."""

    def __init__(self, scale, center, axis, spin, pal, t_birth, dur, n_pet=24, nu=14, nv=22, seed=0):
        rng = np.random.default_rng(seed)
        self.scale, self.center = scale, np.asarray(center, float)
        self.dark, self.mid = pal
        self.t_birth, self.dur = t_birth, dur
        z = unit(axis)
        ref = UP if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
        xa = unit(np.cross(ref, z))
        ya = np.cross(z, xa)
        c, s = math.cos(spin), math.sin(spin)
        self.R = np.stack([xa * c + ya * s, -xa * s + ya * c, z], axis=1)
        self.f = ((np.arange(n_pet) + 0.5) / n_pet)[:, None, None]
        self.u = np.linspace(0.04, 1.0, nu)[None, :, None]
        self.v = np.linspace(-1.0, 1.0, nv)[None, None, :]
        self.a = (np.arange(n_pet) * 2.39996 + rng.normal(0, 0.12, n_pet))[:, None, None]
        self.jit = rng.uniform(0.92, 1.08, (n_pet, 1, 1))
        self.nu, self.nv, self.n_pet = nu, nv, n_pet
        shape = (n_pet, nu, nv)
        self.xp = np.broadcast_to(self.u, shape).ravel().copy()      # 0 base del pétalo ... 1 borde
        self.pr = np.broadcast_to(rng.uniform(0.0, 1.0, (n_pet, 1, 1)), shape).ravel().copy()   # brillo propio de cada pétalo
        self._key, self._P, self._N = None, None, None

    def geometry(self, open_amt):
        key = round(float(open_amt), 3)
        if key == self._key:
            return self._P, self._N
        o = float(open_amt)
        f, u, v = self.f, self.u, self.v
        k = 0.42 + 0.58 * o
        rho0 = 0.05 + 0.34 * f ** 1.1
        rho1 = 0.20 + 0.82 * f ** 1.0
        rho = (rho0 + (rho1 - rho0) * u ** 1.25) * k
        H = (0.86 - 0.40 * f * o - 0.08 * f) * self.jit
        zz = H * u ** 0.9 - 0.10 * v ** 2 * u ** 2 * (0.4 + 0.6 * o)
        zz = zz - 0.22 * o * f ** 1.5 * np.maximum(u - 0.62, 0) ** 1.3 * 2.4   # el borde exterior cae al abrir
        span = (2.6 - 0.8 * f) * (0.50 + 0.50 * np.sqrt(u))
        psi = self.a + v * span / 2
        P = np.stack([rho * np.cos(psi), rho * np.sin(psi), zz + 0 * psi], -1)
        N = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=2))
        N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
        P = P.reshape(-1, 3) * self.scale
        P = P @ self.R.T + self.center
        N = N.reshape(-1, 3) @ self.R.T
        self._key, self._P, self._N = key, P, N
        return P, N


# =====================================================================================
#  El ramo completo
# =====================================================================================
class Bouquet:
    TIE = np.array([0.0, 0.0, -0.66])             # donde se ata el listón
    CLOUD = np.array([0.0, 0.0, 0.42])            # centro de la nube de baby's breath

    def __init__(self, seed=21, n_main=34, max_depth=5, n_roses=6):
        rng = self.rng = np.random.default_rng(seed)
        self.max_depth = max_depth
        self._sp, self._su, self._sk = [], [], []   # tallos: puntos, longitud acumulada, tipo
        self._fl = []                               # flores: centro, normal, radio, u
        self.roses_def = []
        self.rose_c = []
        self.rose_u = []

        # ---------- rosas rosadas: tallo + flor orientada hacia afuera ----------
        polars = [24, 42, 62, 34, 70, 52]
        for j in range(n_roses):
            polar = math.radians(polars[j % len(polars)])
            az = j * (2 * math.pi / n_roses) + 0.35 + rng.normal() * 0.12
            d = np.array([math.sin(polar) * math.cos(az), math.sin(polar) * math.sin(az), math.cos(polar)])
            rad = rng.uniform(0.92, 1.02)
            C = self.CLOUD + d * rad
            scale = rng.uniform(0.28, 0.34)
            self.trunk(C - d * 0.14, d, 0.0, kind=1, depth0=99)
            self.roses_def.append((C, d, scale, rng.uniform(0, 6.28), ROSE_PALETTES[j % 3]))
            self.rose_c.append((C, scale))

        # ---------- baby's breath: cada rama principal se bifurca una y otra vez ----------
        golden = math.pi * (3 - math.sqrt(5))
        cap = math.radians(82)
        for i in range(n_main):
            f = (i + 0.5) / n_main
            polar = math.acos(1 - f * (1 - math.cos(cap)))
            az = i * golden + rng.normal() * 0.08
            d = np.array([math.sin(polar) * math.cos(az), math.sin(polar) * math.sin(az), math.cos(polar)])
            rF = 0.04 + 0.10 * (polar / cap) + rng.uniform(0, 0.05)
            F = np.array([rF * math.cos(az), rF * math.sin(az), rng.uniform(-0.15, 0.30) - 0.25 * (polar / cap)])
            self.trunk(F, d, rng.uniform(0.9, 1.15) * 0.37, kind=0, depth0=0)

        self.finish()
        self.make_ribbon()
        self.make_pollen()

    # ---------------------------------------------------------------- tallos -------
    def add_stem(self, pts, u, kind):
        self._sp.append(pts)
        self._su.append(u)
        self._sk.append(np.full(len(pts), kind))

    def trunk(self, F, d, L0, kind, depth0):
        """Tallo desde la base del ramo hasta F (curva suave) y luego ramas libres."""
        rng = self.rng
        a = rng.uniform(0, 2 * math.pi)
        rr = 0.045 * math.sqrt(rng.random())
        zb = -1.75 - rng.uniform(0, 0.14)
        P0 = np.array([rr * math.cos(a), rr * math.sin(a), zb])
        P1 = np.array([P0[0] * 0.8, P0[1] * 0.8, zb + 0.75])
        P2 = F - d * (0.40 if kind == 0 else 0.30)
        pts, u = resample(bezier(P0, P1, P2, F), DS if kind == 0 else DS * 0.6)
        self.add_stem(pts, u, kind)
        if kind == 1:
            self.rose_u.append(u[-1])
        if kind == 0:
            self.grow(F, d, L0, depth0, u[-1])

    def grow(self, p0, d, L, depth, u0):
        rng = self.rng
        kv = perp(rng, d) * rng.uniform(0.15, 0.55) + UP * 0.10
        K = kv / L
        n = max(3, int(L / DS))
        s = np.linspace(0, L, n)[:, None]
        pts = p0 + d * s + 0.5 * K * s ** 2
        self.add_stem(pts, u0 + s[:, 0], 0)
        end, tend, u_end = pts[-1], unit(d + K * L), u0 + L
        if depth >= self.max_depth:
            self.flower(end, tend, u_end)
            if rng.random() < 0.55:
                self.flower(end - tend * 0.03 + perp(rng, tend) * 0.022, tend, u_end)
            return
        if rng.random() < 0.50:
            self.flower(end + perp(rng, tend) * 0.012, tend, u_end)
        three = rng.random() < 0.20
        ax = perp(rng, tend)
        a = rng.uniform(0.36, 0.70)
        angs = [-a, 0.0, a * rng.uniform(0.8, 1.1)] if three else [-a, a * rng.uniform(0.8, 1.1)]
        for ang in angs:
            if depth >= 3 and rng.random() < 0.14:
                continue
            cd = unit(rot_axis(tend, ax, ang) + UP * 0.12)
            self.grow(end, cd, L * rng.uniform(0.58, 0.72), depth + 1, u_end)

    def flower(self, C, tdir, u):
        rng = self.rng
        for (rc, rs) in self.rose_c:                      # las rosas dejan espacio: sin flores encima
            if np.linalg.norm(C - rc) < rs * 1.15 and rng.random() < 0.92:
                return
        out = unit(C - self.CLOUD)
        n = unit(rng.normal(size=3) * 0.8 + out * 0.9 + tdir * 0.3)
        R = rng.uniform(0.019, 0.031)
        self._fl.append((C, n, R, u))

    # ---------------------------------------------------------------- ensamblado ----
    def finish(self):
        rng = self.rng
        self.sP = np.concatenate(self._sp)
        self.sU = np.concatenate(self._su)
        self.sK = np.concatenate(self._sk)
        fC = np.array([f[0] for f in self._fl])
        fN = np.array([f[1] for f in self._fl])
        fR = np.array([f[2] for f in self._fl])
        fU = np.array([f[3] for f in self._fl])
        self.umax = max(self.sU.max(), fU.max()) * 1.0
        F = len(fC)
        self.F = F
        e1 = unit(np.cross(fN, rng.normal(size=(F, 3))))
        e2 = np.cross(fN, e1)
        self.fC, self.fN, self.fR = fC, fN, fR
        self.f_reach = reach_time(fU, self.umax)
        self.f_t0 = self.f_reach + rng.uniform(0.10, 1.05, F)
        self.f_dur = rng.uniform(1.8, 3.2, F)
        self.f_bmax = np.where(rng.random(F) < 0.30, rng.uniform(0.55, 0.9, F), 1.0)
        tinted = rng.random(F) < 0.20
        self.f_tint = np.where(tinted, rng.uniform(0.35, 0.85, F), 0.0)
        dist = np.linalg.norm(fC - self.CLOUD, axis=1)
        self.f_ao = 0.72 + 0.28 * clamp01((dist - 0.25) / 0.75)   # el interior de la nube queda más a la sombra

        # puntos por flor: 5 pétalos x 8 + 7 del centro
        npet, kp, nc = 5, 8, 7
        per = npet * kp + nc
        fi = np.repeat(np.arange(F), per)
        az0 = np.zeros((F, per))
        tt = np.zeros((F, per))
        ww = np.zeros((F, per))
        hz = np.zeros((F, per))
        kind = np.zeros((F, per))
        spin = rng.uniform(0, 2 * math.pi, F)[:, None]
        for p in range(npet):
            sl = slice(p * kp, (p + 1) * kp)
            az0[:, sl] = (2 * math.pi * p / npet) + rng.normal(0, 0.07, (F, kp))
            tt[:, sl] = 0.10 + 0.90 * np.sqrt(rng.random((F, kp)))
            ww[:, sl] = rng.uniform(-1, 1, (F, kp))
        az0[:, :npet * kp] += spin
        # muesca en la punta del pétalo
        tt[:, :npet * kp] *= 1 - 0.10 * np.exp(-(ww[:, :npet * kp] / 0.35) ** 2)
        cs = slice(npet * kp, per)
        az0[:, cs] = rng.uniform(0, 2 * math.pi, (F, nc))
        tt[:, cs] = 0.30 * np.sqrt(rng.random((F, nc)))
        hz[:, cs] = 0.10 + 0.10 * rng.random((F, nc))
        kind[:, cs] = 1
        self.pf = fi
        self.p_az = az0.ravel()
        self.p_t = tt.ravel()
        self.p_w = ww.ravel()
        self.p_hz = hz.ravel()
        self.p_eye = kind.ravel()
        self.p_ca, self.p_sa = np.cos(self.p_az), np.sin(self.p_az)
        self.p_pw = np.sqrt(self.p_t) * np.sqrt(np.maximum(1 - self.p_t ** 2.5, 0)) * 1.1
        self.p_C, self.p_E1, self.p_E2, self.p_N = fC[fi], e1[fi], e2[fi], fN[fi]
        self.p_R = fR[fi]
        self.p_rnd = rng.random(len(fi))

        # rosas
        self.roses = []
        for (C, d, scale, spin_r, pal) in self.roses_def:
            reach = float(reach_time(self.rose_u[len(self.roses)] + 0.15, self.umax))
            r = Rose(scale, C, d, spin_r, pal, seed=len(self.roses) + 3,
                     t_birth=reach + rng.uniform(0.3, 1.4), dur=rng.uniform(3.4, 4.6))
            r.t_show = reach
            self.roses.append(r)
        self.s_rnd = rng.random(len(self.sP))

    # ---------------------------------------------------------------- listón ---------
    def make_ribbon(self):
        rng = self.rng
        T = self.TIE
        th0 = 0.9                                   # en qué lado del ramo está el moño
        rhat = np.array([math.cos(th0), math.sin(th0), 0.0])
        that = np.array([-math.sin(th0), math.cos(th0), 0.0])
        rb = 0.085
        pts, nrm, sh = [], [], []

        # banda alrededor de los tallos
        th = np.linspace(0, 2 * math.pi, 150, endpoint=False)
        for zz in np.linspace(-0.045, 0.045, 5):
            p = np.stack([rb * np.cos(th), rb * np.sin(th), np.full_like(th, T[2] + zz)], -1)
            pts.append(p)
            nrm.append(np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1))
            sh.append(0.5 + 0.5 * np.sin(th * 3 + 1.0))
        knot = T + rhat * (rb + 0.01)

        def loop(sign):
            a, b, w = 0.17, 0.095, 0.024
            c = knot + that * sign * (a + 0.015) + rhat * 0.025
            out_p, out_n, out_s = [], [], []
            for psi in np.linspace(0, 2 * math.pi, 130, endpoint=False):
                for v in np.linspace(-1, 1, 3):
                    q = (a + v * w) * math.cos(psi) * that * sign * -1 + (b + v * w) * math.sin(psi) * UP
                    out_p.append(c + q + rhat * (0.035 * math.sin(psi) ** 2 + 0.012 * v))
                    out_n.append(unit(rhat + 0.45 * (math.cos(psi) * that * sign * -1 + math.sin(psi) * UP) * v))
                    out_s.append(0.5 + 0.5 * math.sin(2 * psi + (1.3 if sign > 0 else 0.2)))
            return out_p, out_n, out_s
        for sgn in (1, -1):
            p, n, s = loop(sgn)
            pts.append(np.array(p)); nrm.append(np.array(n)); sh.append(np.array(s))

        # nudo
        kn = unit(rng.normal(size=(60, 3)))
        pts.append(knot + kn * np.array([0.030, 0.030, 0.034]) + rhat * 0.02)
        nrm.append(unit(kn + rhat))
        sh.append(rng.random(60))

        # colas
        for sgn, ln, ph in ((1, 0.60, 0.3), (-1, 0.50, 1.8)):
            s = np.linspace(0, ln, 90)
            for v in np.linspace(-1, 1, 4):
                lat = sgn * (0.025 + 0.20 * s) + 0.03 * np.sin(s * 7 + ph) * s * 4 + v * 0.026 * (1 - 0.3 * s / ln)
                p = (knot[None, :] + rhat[None, :] * (0.05 + 0.05 * np.sin(s * 5 + ph))[:, None]
                     + that[None, :] * lat[:, None] + UP[None, :] * (-s * 0.95)[:, None])
                pts.append(p)
                nrm.append(unit(np.tile(rhat, (len(s), 1)) + 0.3 * np.cos(s * 7 + ph)[:, None] * that))
                sh.append(0.5 + 0.5 * np.sin(s * 14 + ph))
        self.rP = np.concatenate(pts)
        self.rN = np.concatenate(nrm)
        self.rS = np.concatenate(sh)
        self.rrnd = rng.random(len(self.rP))

    # ---------------------------------------------------------------- polen ----------
    def make_pollen(self, n=90):
        rng = self.rng
        a = rng.uniform(0, 2 * math.pi, n)
        rad = 1.9 * np.sqrt(rng.random(n))
        self.dP0 = np.stack([rad * np.cos(a), rad * np.sin(a), rng.uniform(-1.8, 2.0, n)], -1)
        self.dPh = rng.uniform(0, 6.28, n)
        self.dRnd = rng.random(n)

    # ---------------------------------------------------------------- por cuadro -----
    def points(self, t):
        """Devuelve P, N, A (albedo), amb, diff, spec, alpha, grupo de cada punto."""
        parts = []
        umax = self.umax

        # --- tallos ---
        g = float(ease_in_out((t - T_START) / T_GROW))
        vis = (self.sU / umax) <= g + 1e-6
        P = self.sP[vis]
        k = self.sK[vis]
        A = np.where((k == 1)[:, None], SAGE_DARK + (SAGE - SAGE_DARK) * 0.3, SAGE)
        A = A * (0.85 + 0.3 * self.s_rnd[vis])[:, None]
        n = len(P)
        parts.append((P, np.tile(UP, (n, 1)), A, 0.55 + 0.25 * self.s_rnd[vis], np.zeros(n), np.zeros(n), np.ones(n), 0))

        # --- flores de baby's breath ---
        F = self.F
        b = ease_in_out((t - self.f_t0) / self.f_dur) * self.f_bmax
        shown = (t >= self.f_reach)
        bp = b[self.pf]
        ok = shown[self.pf]
        phi = 1.18 * (1 - bp) + 0.14
        ext = self.p_R * (0.30 + 0.70 * bp)
        t_ = self.p_t
        rho = ext * t_ * np.cos(phi)
        h = ext * (t_ * np.sin(phi) + 0.10 * t_ * t_ * (1 - bp) + self.p_hz)
        wd = ext * 0.62 * self.p_w * self.p_pw * (0.55 + 0.45 * bp)
        ca, sa = self.p_ca, self.p_sa
        along = (rho * ca - wd * sa)[:, None]
        side = (rho * sa + wd * ca)[:, None]
        P = self.p_C + self.p_E1 * along + self.p_E2 * side + self.p_N * h[:, None]
        tint = self.f_tint[self.pf]
        base = WHITE[None, :] * (1 - tint[:, None]) + BLUSH_TINT[None, :] * tint[:, None]
        eye = clamp01((0.34 - t_) / 0.34)[:, None]
        eye = np.maximum(eye * 0.85, self.p_eye[:, None])
        A = base * (1 - eye) + EYE * eye
        green = (clamp01(1 - bp) ** 1.4 * 0.55)[:, None]
        A = A * (1 - green) + BUD_GREEN * green
        ao = self.f_ao[self.pf]
        amb = (0.58 * ao)
        diff = (0.62 * ao)
        sel = ok
        parts.append((P[sel], self.p_N[sel], A[sel], amb[sel], diff[sel], np.zeros(sel.sum()), np.ones(sel.sum()), 1))

        # --- rosas ---
        for r in self.roses:
            if t < r.t_show:
                continue
            o = float(ease_in_out((t - r.t_birth) / r.dur))
            Pr, Nr = r.geometry(o)
            xp = r.xp
            ao = (0.05 + 0.95 * xp ** 1.7) * (0.80 + 0.35 * r.pr)
            rim = clamp01((xp - 0.76) / 0.24) * (0.7 + 0.6 * r.pr)
            m = np.clip(1 - o, 0, 1) ** 1.5 * 0.45
            mid = r.mid[None, :] * (1 - m) + BUD_GREEN[None, :] * m
            n = len(xp)
            A = np.tile(mid, (n, 1))
            parts.append((Pr, Nr, A, ao, np.zeros(n), rim, np.ones(n), 2, np.tile(r.dark, (n, 1))))

        # --- listón ---
        rs = float(ease_out_cubic((t - T_RIBBON) / 1.3))
        if rs > 0.001:
            P = self.TIE + (self.rP - self.TIE) * rs
            A = RIBBON[None, :] * (0.88 + 0.12 * self.rS[:, None])
            n = len(P)
            parts.append((P, self.rN, A, 0.30 + 0.0 * self.rS, 0.80 * np.ones(n), 0.55 * self.rS ** 2, np.ones(n), 3))

        # --- polen flotando ---
        z = ((self.dP0[:, 2] + 1.8 + 0.09 * t) % 3.8) - 1.8
        P = np.stack([self.dP0[:, 0] + 0.05 * np.sin(0.7 * t + self.dPh),
                      self.dP0[:, 1] + 0.05 * np.cos(0.6 * t + self.dPh), z], -1)
        tw = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2.2 * t + self.dPh * 3))
        A = np.tile(POLLEN, (len(P), 1)) * tw[:, None]
        n = len(P)
        fade = clamp01((t - 3.0) / 1.5)
        parts.append((P, np.tile(UP, (n, 1)), A, np.full(n, 0.9), np.zeros(n), np.zeros(n), np.full(n, fade * 0.8), 4))

        cat = lambda i: np.concatenate([p[i] for p in parts])
        grp = np.concatenate([np.full(len(p[0]), p[7]) for p in parts])
        dk = np.concatenate([p[8] if len(p) > 8 else np.zeros((len(p[0]), 3)) for p in parts])
        return cat(0), cat(1), cat(2), cat(3), cat(4), cat(5), cat(6), grp, dk


# =====================================================================================
#  Renderizado: cada punto 3D cae en una celda de la rejilla y se dibuja con una letra
# =====================================================================================
class Renderer:
    def __init__(self, pygame, W, H):
        self.pg = pygame
        self.W, self.H = W, H
        self.bq = Bouquet()
        fs = max(5, round(W / 170))
        names = "jetbrainsmono,dejavusansmono,consolas,couriernew"
        self.font = pygame.font.SysFont(names, fs, bold=True)
        self.cw = self.font.size("m")[0]
        self.ch = max(1, int(self.font.get_height() * 0.90))
        self.cols, self.rows = W // self.cw + 1, H // self.ch + 1
        self.cap_font = pygame.font.SysFont(names, max(14, round(W / 17)), bold=False)
        self.glyphs = {}

        # Encuadre con el ramo ya abierto
        P, *_ = self.bq.points(9.0)
        sx, up, _ = rotate_view(P[: len(self.bq.sP)], 0.0, ELEV)
        ang_ext = np.sqrt(self.bq.sP[:, 0] ** 2 + self.bq.sP[:, 1] ** 2).max()
        Rxy = max(ang_ext, 1.15)
        _, upf, _ = rotate_view(P, 0.0, ELEV)
        top, bot = upf[: len(self.bq.sP)].max() + 0.1, upf[: len(self.bq.sP)].min()
        self.S0 = min(W * 0.90 / (2 * Rxy), H * 0.72 / (top - bot))
        self.mid = (top + bot) / 2
        self.R = Rxy
        self.cx, self.cy = W / 2, H * 0.44

        L = np.array([-0.35, 0.55, 0.75])
        self.L = L / np.linalg.norm(L)
        self.surf = pygame.Surface((W, H))
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H * 0.45) / (H * 0.62)) ** 2)
        v = clamp01(1 - rr) ** 1.5
        bg = np.stack([5 + 20 * v, 3 + 12 * v, 8 + 24 * v], -1).astype(np.uint8)
        self.bg = pygame.surfarray.make_surface(bg.swapaxes(0, 1))

    def glyph(self, ch, rgb):
        key = (ch, rgb)
        g = self.glyphs.get(key)
        if g is None:
            g = self.font.render(ch, True, rgb)
            self.glyphs[key] = g
        return g

    def frame(self, t):
        pg, bq = self.pg, self.bq
        tl = t % LOOP
        ang = SPIN * tl + 0.4
        elev = ELEV + 0.02 * math.sin(0.9 * tl)
        P, Nn, A, amb, diff, spec, alpha, grp, dk = bq.points(tl)

        sx, up, depth = rotate_view(P, ang, elev)
        nx, nup, ndep = rotate_view(Nn, ang, elev)
        lam = np.abs(self.L[0] * nx + self.L[1] * nup + self.L[2] * ndep) ** 1.2
        rgb = A * (amb + diff * lam)[:, None]
        rgb = rgb + (255 - rgb) * (spec * lam)[:, None]
        ro = grp == 2
        if ro.any():                      # rosas: rampa oscuro -> medio -> claro según la luz y la sombra entre pétalos
            lr = np.abs(self.L[0] * nx[ro] + self.L[1] * nup[ro] + self.L[2] * ndep[ro])
            ao_r, rim_r = amb[ro], spec[ro]
            sh = clamp01((0.06 + 0.85 * lr ** 1.3 + 0.2 * np.abs(ndep[ro])) * ao_r + 0.55 * rim_r * lr)[:, None]
            mid_r, dk_r = A[ro], dk[ro]
            lo = dk_r + (mid_r - dk_r) * clamp01(sh / 0.62)
            hi = mid_r + (ROSE_HI - mid_r) * clamp01((sh - 0.62) / 0.38)
            rgb[ro] = np.where(sh < 0.62, lo, hi)
        dn = clamp01(depth / (2 * self.R) + 0.5)
        rgb = rgb * (0.74 + 0.26 * dn)[:, None]            # lo lejano se apaga un poco (profundidad)

        persp = 1 + 0.10 * (depth / self.R)
        S = self.S0 * persp
        px = self.cx + sx * S
        py = self.cy - (up - self.mid) * S
        age = np.zeros_like(px)

        # Desintegración: el viento sopla hacia la derecha; primero se van los puntos de la derecha
        if tl >= T_DIS:
            sxd, _, _ = rotate_view(P, SPIN * T_DIS + 0.4, ELEV)
            order = clamp01(sxd / (2 * self.R) + 0.5)
            rnd = bq_rnd(bq, len(P), grp)
            r0, r1, r2 = rnd[:, 0], rnd[:, 1], rnd[:, 2]
            t_rel = T_DIS + (1 - order) * 1.5 + r0 * 0.35
            age = np.maximum(tl - t_rel, 0.0)
            wind = (1.1 * age ** 2 + 0.6 * age) * (0.5 + r1)
            px = px + wind * self.S0
            py = py + ((r2 - 0.5) * 0.9 * age - 0.35 * age + 0.05 * np.sin(r2 * 6.283 + age * 4) * age) * self.S0
            keep = r1 < np.where(grp == 0, 0.20, 0.13)         # al soltarse solo queda polvo fino
            alpha = alpha * clamp01(1.25 - age / 1.4) * np.where((age > 0.04) & ~keep, 0.0, 1.0)
            rgb = rgb + (DUST - rgb) * (clamp01(age * 3) * 0.55)[:, None]
        rgb = rgb * alpha[:, None]

        col = (px / self.cw).astype(np.int64)
        row = (py / self.ch).astype(np.int64)
        ok = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows) & (alpha > 0.02)
        idx = (row * self.cols + col)[ok]
        surf = self.surf
        surf.blit(self.bg, (0, 0))
        if idx.size:
            pts = np.nonzero(ok)[0]
            o = np.lexsort((depth[ok], idx))
            last = np.r_[idx[o][1:] != idx[o][:-1], True]
            sel = pts[o[last]]
            cells = idx[o[last]]
            q = (np.clip(rgb[sel], 0, 255) // 8 * 8).astype(np.int64)
            cols, rows, cw, ch = self.cols, self.rows, self.cw, self.ch
            n = len(PHRASE)
            blit = surf.blit
            for cell, (r, g, b) in zip(cells.tolist(), q.tolist()):
                if r + g + b < 30:
                    continue
                cr, cc = divmod(cell, cols)
                c = PHRASE[(cc + cr * 5) % n]
                if c == " ":
                    continue
                blit(self.glyph(c, (r, g, b)), (cc * cw, cr * ch))

        self.bloom(surf)
        self.caption(surf, tl)
        return surf

    def caption(self, surf, tl):
        a = float(clamp01((tl - 8.2) / 1.2)) * float(clamp01((T_DIS + 0.9 - tl) / 0.9))
        if a <= 0.01:
            return
        img = self.cap_font.render(CAPTION, True, (252, 226, 234))
        img.set_alpha(int(220 * a))
        surf.blit(img, ((self.W - img.get_width()) // 2, int(self.H * 0.865)))

    def bloom(self, surf):
        pg = self.pg
        W, H = self.W, self.H
        s1 = pg.transform.smoothscale(surf, (max(1, W // 4), max(1, H // 4)))
        s2 = pg.transform.smoothscale(s1, (max(1, W // 16), max(1, H // 16)))
        g1 = pg.transform.smoothscale(s1, (W, H))
        g1.fill((120, 120, 120), special_flags=pg.BLEND_RGB_MULT)
        g2 = pg.transform.smoothscale(s2, (W, H))
        g2.fill((150, 150, 150), special_flags=pg.BLEND_RGB_MULT)
        surf.blit(g1, (0, 0), special_flags=pg.BLEND_RGB_ADD)
        surf.blit(g2, (0, 0), special_flags=pg.BLEND_RGB_ADD)


_RND_CACHE = {}


def bq_rnd(bq, n, grp):
    """Números aleatorios fijos por punto (se reutilizan mientras el número de puntos no cambie)."""
    key = n
    r = _RND_CACHE.get(key)
    if r is None:
        r = np.random.default_rng(5).random((n, 3))
        _RND_CACHE.clear()
        _RND_CACHE[key] = r
    return r


# =====================================================================================
def run_window(W, H):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("forever my favorite")
    rend = Renderer(pygame, W, H)
    clock = pygame.time.Clock()
    t, paused = 0.0, False
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                t = 0.0
            if e.type == pygame.VIDEORESIZE:
                rend = Renderer(pygame, e.w, e.h)
        dt = clock.tick(60) / 1000
        if not paused:
            t += dt
        screen.blit(rend.frame(t), (0, 0))
        pygame.display.flip()


def run_still(path, W, H, t):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1, 1))
    rend = Renderer(pygame, W, H)
    pygame.image.save(rend.frame(t), path)
    print(f"listo: {path}", file=sys.stderr)


def run_record(path, W, H, fps, loops):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1, 1))
    rend = Renderer(pygame, W, H)
    total = int(round(LOOP * loops * fps))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
         "-movflags", "+faststart", path],
        stdin=subprocess.PIPE)
    for i in range(total):
        surf = rend.frame(i / fps)
        ff.stdin.write(pygame.image.tobytes(surf, "RGB"))
        if i % fps == 0:
            print(f"\r{i}/{total}", end="", file=sys.stderr, flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"\rlisto: {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", metavar="MP4", help="exporta a video en vez de abrir ventana")
    ap.add_argument("--still", nargs=2, metavar=("SEG", "PNG"), help="guarda un solo cuadro como imagen")
    ap.add_argument("--size", default=None, help="ANCHOxALTO (ventana 900x1200, video 1080x1920)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--loops", type=float, default=1.0)
    a = ap.parse_args()
    size = a.size or ("1080x1920" if (a.record or a.still) else "900x1200")
    W, H = (int(v) for v in size.lower().split("x"))
    if a.still:
        run_still(a.still[1], W, H, float(a.still[0]))
    elif a.record:
        run_record(a.record, W, H, a.fps, a.loops)
    else:
        run_window(W, H)


if __name__ == "__main__":
    main()
