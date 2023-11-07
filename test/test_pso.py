from neural_swarm.activation.relu import Relu
from neural_swarm.pso.particle import Particle
from test.test_data import prep_test_ann


def test_particle_encode_decode():
    ann = prep_test_ann()
    particle = Particle(ann)

    print("Initial Weights: ")
    for layer in ann.network.layers:
        if layer.weights is not None:
            print(layer.weights)

    encoded = particle.encode()
    print("\nEncoded Result: ", encoded)
    print("\nDecoded Result: ", particle.decode(encoded))
