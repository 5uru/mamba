import jax
import jax.numpy as jnp
from jax.lax import associative_scan
from jax import lax

# ==============================================================================
# VRAI PARALLEL SCAN (Associative Scan)
# ==============================================================================
def pscan_jax(A, X):
    """
    Implémentation Parallel Scan optimisée pour JAX.
    Complexité temporelle : O(L * log(L)) au lieu de O(L^2) ou O(L) séquentiel.

    A : (B, L, D, N) - Matrices de transition (diagonales)
    X : (B, L, D, N) - Vecteurs d'entrée
    Retourne : (B, L, D, N) - États cachés
    """
    # L'opération binaire associative pour le SSM :
    # Si on a deux pas de temps (a1, b1) et (a2, b2) tels que :
    # h1 = a1 * h0 + b1
    # h2 = a2 * h1 + b2
    # Alors les combiner donne : h2 = (a2*a1)*h0 + (a2*b1 + b2)
    # Comme A est diagonale, la multiplication matricielle est élément-wise (*)
    def combine(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return a2 * a1, a2 * b1 + b2

    # associative_scan travaille sur l'axe 0. On déplace la dimension L (axe 1) à l'axe 0.
    A_transposed = jnp.moveaxis(A, 1, 0)  # Shape : (L, B, D, N)
    X_transposed = jnp.moveaxis(X, 1, 0)  # Shape : (L, B, D, N)

    # Exécution du scan parallèle
    _, Y_transposed = jax.lax.associative_scan(combine, (A_transposed, X_transposed), axis=0)

    # On remet L à sa place initiale (axe 0 -> axe 1)
    return jnp.moveaxis(Y_transposed, 0, 1)


