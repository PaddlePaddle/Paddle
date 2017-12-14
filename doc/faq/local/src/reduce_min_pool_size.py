@provider(min_pool_size=0, ...)
def process(settings, filename):
    os.system('shuf %s > %s.shuf' % (filename, filename))  # shuffle before.
    with open('%s.shuf' % filename, 'r') as f:
        for line in f:
            yield get_sample_from_line(line)
