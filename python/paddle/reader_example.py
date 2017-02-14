# We will look at 4 ways to create iterative function, and 2 ways to create reader

# there are 4 ways to build iterators in python (http://stackoverflow.com/questions/19151/build-a-basic-python-iterator)
# Following are 4 ways to build iterators 


# 13 lines
class using_class():
    def __init__(self):
        self.counter = 0

    def __iter__(self):
        self.counter = 0
        return self

    def next(self):
        if self.counter > 1:
            raise StopIteration
        ret = self.counter
        self.counter = self.counter + 1
        return ret


# 10 lines
def using_yield():
    class nonlocal:  # this hack is for python 2, because closure variables can not be mutated in Python 2
        counter = 0

    def iter():
        while nonlocal.counter < 2:
            ret = nonlocal.counter
            nonlocal.counter = nonlocal.counter + 1
            yield ret

    return iter()


# 2 lines
def using_generator():
    return (i for i in xrange(2))


# 9 lines
class using_get_item():
    def __init__(self):
        self.counter = 0

    def __getitem__(self, index):
        if self.counter > 1:
            raise StopIteration
        ret = self.counter
        self.counter = self.counter + 1
        return ret


# Following are 2 ways to create reader:


# 13 lines
def create_reader_closure():
    class nonlocal:  # this hack is for python 2, because closure variables can not be mutated in Python 2
        counter = 0

    def read(reset):
        if reset:
            nonlocal.counter = 0
        if nonlocal.counter > 1:
            return None
        ret = nonlocal.counter
        nonlocal.counter = nonlocal.counter + 1
        return ret

    return read


# 11 lines
class create_reader_class():
    def __init__(self):
        self.counter = 0

    def read(self, reset):
        if reset:
            self.counter = 0
        if self.counter > 1:
            return None
        ret = self.counter
        self.counter = self.counter + 1
        return ret


# Following are way to use iterator and way to use reader:


# 7 lines
def evaluate_iterator(f, name, total_pass):
    print "evaluating", name
    total = 0
    while total < total_pass:
        print "one pass begin"
        for mini_batch in f:
            print mini_batch
        total = total + 1


# 9 lines
def evaluate_reader(f, name, total_pass):
    print "evaluating", name
    total = 0
    while total < total_pass:
        print "one pass begin"
        mini_batch = f(True)
        while mini_batch is not None:
            print mini_batch
            mini_batch = f(False)
        total = total + 1


# outputs:
# evaluating class
# one pass begin
# 0
# 1
# one pass begin
# 0
# 1
evaluate_iterator(using_class(), "class", 2)  # can iterate multiple pass

# outputs:
# evaluating yield
# one pass begin
# 0
# 1
# one pass begin
evaluate_iterator(using_yield(), "yield", 2)  # can NOT iterate multiple pass

# outputs:
# evaluating generator
# one pass begin
# 0
# 1
# one pass begin
evaluate_iterator(using_generator(), "generator",
                  2)  # can NOT iterate multiple pass

# outputs:
# evaluating get_item
# one pass begin
# 0
# 1
# one pass begin
evaluate_iterator(using_get_item(), "get_item",
                  2)  # can NOT iterate multiple pass

# outputs:
# evaluating reader_closure
# one pass begin
# 0
# 1
# one pass begin
# 0
# 1
evaluate_reader(create_reader_closure(), "reader_closure",
                2)  # can iterate multiple pass

# outputs:
# evaluating reader_class
# one pass begin
# 0
# 1
# one pass begin
# 0
# 1
evaluate_reader(create_reader_class().read, "reader_class",
                2)  # can iterate multiple pass

# Out of 4 ways of creating iterator, only one way (creating a class) can iterate multiple passes.

# Now our options:
# 1. Create an iterator class (and tell user don't use other kind of iterators, since they can't do multiple pass)
#    Also, iterator class require user to implement next in python 2 and __next__ in python 3.
# 2. Create a function with sigature read(reset), which returns a mini batch every time it's called. Returns None if current pass is finished.
#    User can implement the function by closure (create_reader_closure) or class (create_reader_class).
# Personally I like option 2.
