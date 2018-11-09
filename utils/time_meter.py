import time


time_stamp = list()


def measure(text=''):
    """
    Time measure.
    :param str text: Text to display.
    :rtype: None.
    """
    to_display = text + ' '*(30-len(text))
    curr = time.clock()
    if len(time_stamp) > 0:
        to_display = '%s%.4f' % (to_display, curr-time_stamp[-1])
    time_stamp.append(curr)
    to_display = to_display + ' '*(40-len(to_display))
    to_display = to_display + '%.4f' % curr
    time.sleep(0.01)
    print(to_display)
    time.sleep(0.01)


# measure('BEGIN')
