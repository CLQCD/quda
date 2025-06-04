#include <color_spinor_field_order.h>
#include <kernel.h>

#define PRESERVE_SPINOR_NORM

#ifdef PRESERVE_SPINOR_NORM // Preserve the norm regardless of basis
#define kP (1.0 / sqrt(2.0))
#define kU (1.0 / sqrt(2.0))
#else // More numerically accurate not to preserve the norm between basis
#define kP (0.5)
#define kU (1.0)
#endif

namespace quda
{

  using namespace colorspinor;

  /** Straight copy with no basis change */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct PreserveBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      const int offset = (Chirality == QUDA_CHIRALITY_UPPER) ? 0 : Ns / 2;
      if constexpr (Project) {
        for (int s = 0; s < Ns / 2; s++)
          for (int c = 0; c < Nc; c++) out[s * Nc + c] = in[(offset + s) * Nc + c];
      } else {
        for (int s = 0; s < Ns / 2; s++)
          for (int c = 0; c < Nc; c++) out[(offset + s) * Nc + c] = in[s * Nc + c];
      }
    }
  };

  /** Transform from relativistic Degrand-Rossi into non-relativistic UKQCD basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct NonRelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {1, 2, 3, 0};
        int s2[4] = {3, 0, 1, 2};
        FloatOut K1[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP),
                          static_cast<FloatOut>(-kP)};
        FloatOut K2[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP),
                          static_cast<FloatOut>(kP)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c])
              + K2[offset + s] * static_cast<complex<FloatOut>>(in[s2[offset + s] * Nc + c]);
          }
        }
      } else {
        int s1[4] = {1, 0, 1, 0};
        FloatOut K1[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP),
                          upper ? static_cast<FloatOut>(kP) : static_cast<FloatOut>(-kP),
                          upper ? static_cast<FloatOut>(-kP) : static_cast<FloatOut>(kP)};
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
        }
      }
    }
  };

  /** Embed from relativistic Degrand-Rossi into non-relativistic UKQCD basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct NonRelBasisEmbed {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[2 * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      int s1[4] = {1, 0, 1, 0};
      FloatOut K1[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP),
                        upper ? static_cast<FloatOut>(kP) : static_cast<FloatOut>(-kP),
                        upper ? static_cast<FloatOut>(-kP) : static_cast<FloatOut>(kP)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
      }
    }
  };

  /** Transform from non-relativistic UKQCD into relativistic Degrand-Rossi basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct RelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {1, 2, 3, 0};
        int s2[4] = {3, 0, 1, 2};
        FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU),
                          static_cast<FloatOut>(kU)};
        FloatOut K2[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(-kU),
                          static_cast<FloatOut>(-kU)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c])
              + K2[offset + s] * static_cast<complex<FloatOut>>(in[s2[offset + s] * Nc + c]);
            // out[s * Nc + c] = static_cast<FloatOut>(3) * static_cast<complex<FloatOut>>(out[s * Nc + c]);
          }
        }
      } else {
        int s1[4] = {1, 0, 1, 0};
        FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU),
                          upper ? static_cast<FloatOut>(-kU) : static_cast<FloatOut>(kU),
                          upper ? static_cast<FloatOut>(kU) : static_cast<FloatOut>(-kU)};
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
        }
      }
    }
  };

  /** Projection from non-relativistic UKQCD into relativistic Degrand-Rossi basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct RelBasisProj {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[2 * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {1, 2, 3, 0};
        int s2[4] = {3, 0, 1, 2};
        FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU),
                          static_cast<FloatOut>(kU)};
        FloatOut K2[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(-kU),
                          static_cast<FloatOut>(-kU)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c])
              + K2[offset + s] * static_cast<complex<FloatOut>>(in[s2[offset + s] * Nc + c]);
          }
        }
      } 
    }
  };

  /** Transform from non-relativistic Dirac-Pauli into relativistic Degrand-Rossi basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct DegrandRossiToDiracPaulBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {1, 2, 1, 0};
        int s2[4] = {3, 0, 3, 2};
        FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU),
                          static_cast<FloatOut>(-kU)};
        FloatOut K2[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(-kU),
                          static_cast<FloatOut>(kU)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c])
              + K2[offset + s] * static_cast<complex<FloatOut>>(in[s2[offset + s] * Nc + c]);
          }
        }
      } else {
        int s1[4] = {1, 0, 1, 0};
        FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU),
                          upper ? static_cast<FloatOut>(kU) : static_cast<FloatOut>(-kU),
                          upper ? static_cast<FloatOut>(-kU) : static_cast<FloatOut>(kU)};
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
        }
      }
    }
  };

  /** Transform from relativistic Degrand-Rossi into non-relativistic Dirac-Pauli basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct DiracPaulToDegrandRossiBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {1, 2, 1, 0};
        int s2[4] = {3, 0, 3, 2};
        FloatOut K1[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP),
                          static_cast<FloatOut>(-kP)};
        FloatOut K2[4] = {static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP),
                          static_cast<FloatOut>(-kP)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c])
              + K2[offset + s] * static_cast<complex<FloatOut>>(in[s2[offset + s] * Nc + c]);
          }
        }
      } else {
        int s1[4] = {1, 0, 1, 0};
        FloatOut K1[4] = {upper ? static_cast<FloatOut>(kP) : static_cast<FloatOut>(-kP),
                          upper ? static_cast<FloatOut>(-kP) : static_cast<FloatOut>(kP), static_cast<FloatOut>(kP),
                          static_cast<FloatOut>(-kP)};
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
        }
      }
    }
  };

  /** Transform from chiral into UKQCD non-relativistic basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct ChiralToNonRelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {0, 1, 0, 1};
        int s2[4] = {2, 3, 2, 3};
        FloatOut K1[4] = {static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP),
                          static_cast<FloatOut>(kP)};
        FloatOut K2[4]
          = {static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c])
              + K2[offset + s] * static_cast<complex<FloatOut>>(in[s2[offset + s] * Nc + c]);
          }
        }
      } else {
        int s1[4] = {0, 1, 0, 1};
        FloatOut K1[4] = {upper ? static_cast<FloatOut>(-kP) : static_cast<FloatOut>(kP),
                          upper ? static_cast<FloatOut>(-kP) : static_cast<FloatOut>(kP), static_cast<FloatOut>(kP),
                          static_cast<FloatOut>(kP)};
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
        }
      }
    }
  };

  /** Transform from UKQCD non-relativistic into chiral basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct NonRelToChiralBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {0, 1, 0, 1};
        int s2[4] = {2, 3, 2, 3};
        FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU),
                          static_cast<FloatOut>(kU)};
        FloatOut K2[4]
          = {static_cast<FloatOut>(kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c])
              + K2[offset + s] * static_cast<complex<FloatOut>>(in[s2[offset + s] * Nc + c]);
          }
        }
      } else {
        int s1[4] = {0, 1, 0, 1};
        FloatOut K1[4] = {upper ? static_cast<FloatOut>(-kU) : static_cast<FloatOut>(kU),
                          upper ? static_cast<FloatOut>(-kU) : static_cast<FloatOut>(kU), static_cast<FloatOut>(kU),
                          static_cast<FloatOut>(kU)};
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
        }
      }
    }
  };

  /** Transform from chiral into DeGrand-Rossi basis or from DeGrand-Rossi into chiral basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct ChiralToFromDegrandRossiBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {3, 2, 1, 0};
        FloatOut K1[4] = {static_cast<FloatOut>(-1.0), static_cast<FloatOut>(1.0), static_cast<FloatOut>(1.0),
                          static_cast<FloatOut>(-1.0)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c]);
          }
        }
      } else {
        int s1[4] = {1, 0, 1, 0};
        FloatOut K1[4] = {upper ? static_cast<FloatOut>(0.0) : static_cast<FloatOut>(-1.0),
                          upper ? static_cast<FloatOut>(0.0) : static_cast<FloatOut>(1.0),
                          upper ? static_cast<FloatOut>(1.0) : static_cast<FloatOut>(0.0),
                          upper ? static_cast<FloatOut>(-1.0) : static_cast<FloatOut>(0.0)};
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
        }
      }
    }
  };

  /** Transform from UKQCD to Dirac-Pauli and from Dirac-Pauli into UKQCD basis */
  template <int Ns, int Nc, QudaChirality Chirality, bool Project> struct UKQCDToFromDiracPauliBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      constexpr bool upper = (Chirality == QUDA_CHIRALITY_UPPER);
      constexpr int offset = upper ? 0 : Ns / 2;
      if constexpr (Project) {
        int s1[4] = {0, 1, 2, 3};
        FloatOut K1[4] = {static_cast<FloatOut>(-1.0), static_cast<FloatOut>(-1.0), static_cast<FloatOut>(1.0),
                          static_cast<FloatOut>(1.0)};
        for (int s = 0; s < Ns / 2; s++) {
          for (int c = 0; c < Nc; c++) {
            out[s * Nc + c] = K1[offset + s] * static_cast<complex<FloatOut>>(in[s1[offset + s] * Nc + c]);
          }
        }
      } else {
        int s1[4] = {0, 1, 0, 1};
        FloatOut K1[4] = {upper ? static_cast<FloatOut>(-1.0) : static_cast<FloatOut>(0.0),
                          upper ? static_cast<FloatOut>(-1.0) : static_cast<FloatOut>(0.0),
                          upper ? static_cast<FloatOut>(0.0) : static_cast<FloatOut>(1.0),
                          upper ? static_cast<FloatOut>(0.0) : static_cast<FloatOut>(1.0)};
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
        }
      }
    }
  };

} // namespace quda
