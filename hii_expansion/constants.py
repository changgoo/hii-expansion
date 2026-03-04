"""Physical constants in CGS units, sourced from astropy."""

from astropy import units as u
from astropy.constants import k_B, m_p

M_H: float = m_p.cgs.value          # hydrogen atom mass [g]
K_B: float = k_B.cgs.value          # Boltzmann constant [erg K⁻¹]
PC: float = (1.0 * u.pc).cgs.value  # parsec [cm]
YR: float = (1.0 * u.yr).cgs.value  # Julian year [s]
