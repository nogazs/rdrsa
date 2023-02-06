import numpy as np
from prag_models import Lexicon, RSA, RDRSA


# define a lexicon (this example is based on Vogel et al., 2014)
err_prob = 0.01
lex_arr = np.array([[1., 1., err_prob],
                    [err_prob, 1., 1.],
                    [err_prob, err_prob, 1.]])
lex = Lexicon(lex_arr,  # lexicon
              M_labels=['M', 'GM', 'HG'],  # referent labels
              U_labels=['mustache', 'glasses', 'hat']  # utterances
              )

# init models
rdrsa = RDRSA(lex)
rsa = RSA(lex)

# run models until convergence
rdrsa_trajectory = rdrsa.solve(alpha=1.3)
rsa_trajectory = rsa.solve(alpha=1.3)

# print results for recursion depth t=5
print('\nboth models exhibit scalar implicature before convergence:')
rdrsa_trajectory.display_state(t=5)
rsa_trajectory.display_state(t=5)

print('\n======')
print('adding an utterance cost function')

# define an utterance cost function
C = np.array([0.6, 0.2, 0.2])

# init models
rdrsa = RDRSA(lex, C=C)
rsa = RSA(lex, C=C)

# run models until convergence
rdrsa_trajectory = rdrsa.solve(alpha=0.8)
rsa_trajectory = rsa.solve(alpha=0.8)

# print results
print('\nRSA, but not RD-RSA, is biased toward random utterance production even when it\'s not informative:')
rdrsa_trajectory.display_state()
rsa_trajectory.display_state()
