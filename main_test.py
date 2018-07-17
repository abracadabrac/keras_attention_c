from loader import load_xp_model
from Run import test_model


name = '2018-07-10-16-37-16'
net = load_xp_model(name)
test_model(net, name)


