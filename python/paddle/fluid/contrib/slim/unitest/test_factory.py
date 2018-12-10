from paddle.fluid.contrib.slim import ConfigFactory
import unittest

class TestFactory(unittest.TestCase):
    def test_parse(self):
        factory = ConfigFactory('./unitest/configs/config.yaml')

        pruner = factory.instance('pruner_1')
        self.assertEquals(pruner.ratios['conv1_1.w'], 0.3)

        pruner = factory.instance('pruner_2')
        self.assertEquals(pruner.ratios['*'], 0.7)

        strategy = factory.instance('strategy_1')
        pruner = strategy.pruner
        self.assertEquals(pruner.ratios['*'], 0.7)

        compress_pass = factory.get_compress_pass()
        self.assertEquals(compress_pass.epoch, 100)

        strategy = compress_pass.strategies[0]
        self.assertEquals(strategy.delta_rate, 0.2)

if __name__ == '__main__':
    unittest.main() 
