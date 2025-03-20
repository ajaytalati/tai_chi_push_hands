# tai_chi_push_hands
Some simple experiments trying to simulate tai chi "push hands"

# TODO

- Add ego centric camera so that agent can use vision, as well as touch/contact and proprioception - use the example code - https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1_t4.ipynb#scrollTo=06333cd4
- Finish adding contact stuff to xml geom upper body parts for the humanoid, (i.e. contype="1" conaffinity="1") At the moment it is walking throught the punchbag :(    
- Debug - pendulum CoM perturbation reward, (to simulate trying to control/take opponents CoM in tai chi push hands) - need to understand why it's not working?
