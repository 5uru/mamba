import jax
import jax.numpy as jnp
from jax.lax import associative_scan

def pscan_jax(A, X):
    def combine(elem1, elem2):
        A1, X1 = elem1
        A2, X2 = elem2
        return A2 * A1, A2 * X1 + X2

    A_padded = jnp.concatenate([jnp.ones_like(A[:, :1]), A], axis=1)
    X_padded = jnp.concatenate([jnp.zeros_like(X[:, :1]), X], axis=1)

    _, H_padded = associative_scan(combine, (A_padded, X_padded), axis=1)
    return H_padded[:, 1:]

# 1. Définition d'une fonction de perte quelconque
def loss_fn(A, X):
    H = pscan_jax(A, X)
    return jnp.mean(H ** 2) # Loss simple : moyenne des carrés

# 2. Création de données fictives (Batch=2, L=5, D=3, N=4)
key = jax.random.PRNGKey(42)
A_dummy = jax.random.normal(key, (10000, 5, 3, 4))
X_dummy = jax.random.normal(key, (10000, 5, 3, 4))

# 3. Calcul de la loss
loss = loss_fn(A_dummy, X_dummy)
print(f"Loss: {loss}")

# 4. Calcul DES GRADIENTS (En une seule ligne !)
grad_A, grad_X = jax.grad(loss_fn, argnums=(0, 1))(A_dummy, X_dummy)

print(f"Gradient shape pour A: {grad_A.shape}") # (2, 5, 3, 4)
print(f"Gradient shape pour X: {grad_X.shape}") # (2, 5, 3, 4)