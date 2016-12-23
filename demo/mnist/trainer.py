import functools

__all__ = ['RunnerChainItem', 'Runner']


class RunnerChainItem(object):
    def __init__(self):
        pass

    def initialize(self, context, next_callback):
        next_callback(context)

    def finalize(self, context, next_callback):
        next_callback(context)

    def on_pass_begin(self, context, next_callback):
        next_callback(context)

    def on_pass_end(self, context, next_callback):
        next_callback(context)

    def on_batch_begin(self, context, next_callback):
        return next_callback(context)

    def on_batch_end(self, context, next_callback):
        return next_callback(context)


def default_next_callback(*args, **kwargs):
    return False


class RunnerContext(object):
    pass


class RunnerSection(object):
    def __init__(self, runner):
        self.runner = runner

    def __enter__(self):
        self.runner.initialize()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runner.finalize()


class Runner(object):
    def __init__(self):
        self.chains = []

        self.begin_pass = None
        self.end_pass = None
        self.begin_batch = None
        self.end_batch = None
        self.finalize = None

        self.context = RunnerContext()
        self.context.runner = self

    def add_chain_item(self, item):
        assert isinstance(item, RunnerChainItem)
        self.chains.append(item)

    def initialize(self):
        if None not in [
                self.begin_pass, self.end_pass, self.begin_batch,
                self.end_batch, self.finalize
        ]:
            return False
        else:
            assert len(self.chains) != 0
            actual_init = default_next_callback
            self.begin_pass = default_next_callback
            self.end_pass = default_next_callback
            self.begin_batch = default_next_callback
            self.end_batch = default_next_callback
            self.finalize = default_next_callback

            for chain in reversed(self.chains):
                assert isinstance(chain, RunnerChainItem)
                actual_init = functools.partial(
                    chain.initialize, next_callback=actual_init)
                self.begin_pass = functools.partial(
                    chain.on_pass_begin, next_callback=self.begin_pass)
                self.end_pass = functools.partial(
                    chain.on_pass_end, next_callback=self.end_pass)
                self.begin_batch = functools.partial(
                    chain.on_batch_begin, next_callback=self.begin_batch)
                self.end_batch = functools.partial(
                    chain.on_batch_end, next_callback=self.end_batch)
                self.finalize = functools.partial(
                    chain.finalize, next_callback=self.finalize)

            actual_init(self.context)
            return True

    def run_one_pass(self):
        self.begin_pass(self.context)
        exit_flag = False
        while not exit_flag:
            exit_flag = self.begin_batch(self.context)
            if exit_flag:
                break
            exit_flag = self.end_batch(self.context)
        self.end_pass(self.context)

    def use(self):
        return RunnerSection(self)
