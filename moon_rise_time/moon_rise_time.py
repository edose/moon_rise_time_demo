""" Module moon_rise_time.moon_rise_time
    Demonstration of proposed new moon_rise_time() for astroplan.
"""

__author__ = "Eric Dose, Albuquerque"


# Python core:
import os
from math import sqrt, ceil
from time import perf_counter_ns
from random import seed, uniform

# External packages:
from astropy import units as u
from astropy.time import Time, TimeDelta
import astroplan


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def moon_transit_time(observer, time, which='nearest'):
    """ Get moon transit time at observer location. This fn is ABSENT from astroplan for some reason.
    :param observer: astroplan.Observer object.
    :param time:`~astropy.time.Time` or other (see below).
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)
    :param which: {'next', 'previous', 'nearest'}
            Choose which moon transit time relative to the present ``time`` would you
            like to calculate
    :return:`~astropy.time.Time` Time of moon transit
    """
    def metric_fn(time):
        """ Moon transit occurs when this function passes through zero in the positive direction. """
        return observer.moon_altaz(time).az.deg - 180.0  # returns float.

    return _find_best_crossing(time, which, metric_fn, fn_derivative_sign='pos',
                               bracket_duration=2 * u.hour, n_brackets_nearest=7, n_brackets_next=14)


def moon_rise_time(observer, time, which='nearest', horizon=0 * u.deg):
    """ Get moon rise time at observer location. This fn is proposed to REPLACE current astroplan fn.
    :param observer: astroplan.Observer object.
    :param time:`~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)
    :param which: {'next', 'previous', 'nearest'}
            Choose which moon rise relative to the present ``time`` would you
            like to calculate
    :param horizon: Quantity (optional), default = zero degrees
            Degrees above/below actual horizon to use for calculating rise/set times
            (i.e., -6 deg horizon = civil twilight, etc.)
    :return:`~astropy.time.Time` Time of moon rise
    """
    horizon_deg = horizon.value

    def metric_fn(time):
        """ Moon rise occurs when this function passes through zero in the positive direction. """
        return observer.moon_altaz(time).alt.deg - horizon_deg  # returns float.

    return _find_best_crossing(time, which, metric_fn, fn_derivative_sign='pos',
                               bracket_duration=2 * u.hour, n_brackets_nearest=7, n_brackets_next=14)


def _find_best_crossing(time, which, metric_fn, fn_derivative_sign, bracket_duration=2 * u.hour,
                        n_brackets_nearest=7, n_brackets_next=14):
    """ Find crossing (in either time direction) of metric_fn through zero
        (with fn_derivative_sign respected if given).
    :param time:
    :param which:
    :param metric_fn:
    :param fn_derivative_sign:
    :param bracket_duration:
    :param n_brackets_nearest:
    :param n_brackets_next:
    :return:
    """
    if which not in ('next', 'previous', 'nearest'):
        raise ValueError('Parameter \'which\' must be in {\'next\', \'previous\', or \'nearest\'}.')

    if which == 'next':
        i, times, fn_values = _find_next_crossing(time, metric_fn, fn_derivative_sign,
                                                  bracket_duration, n_brackets=n_brackets_next)
        return _refine_event_time(i, times, fn_values, metric_fn)

    elif which == 'previous':
        i, times, fn_values = _find_previous_crossing(time, metric_fn, fn_derivative_sign,
                                                      bracket_duration, n_brackets=n_brackets_next)
        return _refine_event_time(i, times, fn_values, metric_fn)

    elif which == 'nearest':
        val = _find_next_crossing(time, metric_fn, fn_derivative_sign,
                                  bracket_duration, n_brackets=n_brackets_nearest)
        i_next, next_times, next_fn_values = val
        val = _find_previous_crossing(time, metric_fn, fn_derivative_sign,
                                      bracket_duration, n_brackets=n_brackets_nearest)
        i_previous, previous_times, previous_fn_values = val

        # Cases: the search for 'next' or 'previous' (or both) found no crossings:
        if i_next is None and i_previous is None:
            # print('########## NO CROSSING FOUND (should never happen).')
            return None
        if i_previous is None:
            # print('No previous crossing, refine next only:')
            return _refine_event_time(i_next, next_times, next_fn_values, metric_fn)
        if i_next is None:
            # print('No next crossing, refine previous only:')
            return _refine_event_time(i_previous, previous_times, previous_fn_values, metric_fn)

        # Cases: either 'next' and 'previous' search found a crossing.
        # One bracket is clearly nearer than the other, so refine and return.
        if i_previous < i_next:
            return _refine_event_time(i_previous, previous_times, previous_fn_values, metric_fn)
        if i_next < i_previous:
            return _refine_event_time(i_next, next_times, next_fn_values, metric_fn)

        # Remaining case: 'next' and 'previous' brackets appear equally near.
        # Refine both, and return the nearer refined time.
        # print('refine both next and previous:')
        next_time = _refine_event_time(i_next, next_times, next_fn_values, metric_fn)
        previous_time = _refine_event_time(i_previous, previous_times, previous_fn_values, metric_fn)
        if abs(time - previous_time) >= abs(next_time - time):
            return next_time
        return previous_time


def _find_next_crossing(time, metric_fn, fn_derivative_sign, bracket_duration, n_brackets):
    """ Find next crossing of metric_fn through zero (with fn_derivative_sign respected if given).
    :param time:
    :param metric_fn:
    :param fn_derivative_sign: 'pos' or 'neg':
    :param bracket_duration:
    :param n_brackets:
    :return: 2-tuple of Times bracketing the crossing, or None if no crossing found.
    """
    times = [time + i * bracket_duration for i in range(n_brackets + 1)]  # times[0] == time.
    fn_values = metric_fn(times)
    if fn_derivative_sign == 'pos':
        has_crossing = [fn_values[i] <= 0 and fn_values[i + 1] > 0 for i in range(len(times) - 1)]
    elif fn_derivative_sign == 'neg':
        has_crossing = [fn_values[i] > 0 and fn_values[i + 1] <= 0 for i in range(len(times) - 1)]
    else:
        ValueError('fn_derivative_sign must be \'pos\' or \'neg\'.')
    if has_crossing.count(True) <= 0:
        return None, None, None
    i_found = has_crossing.index(True)
    return i_found, times, fn_values


def _find_previous_crossing(time, metric_fn, fn_derivative_sign, bracket_duration, n_brackets):
    """ Find next crossing of metric_fn through zero (with fn_derivative_sign respected if given).
    :param time:
    :param metric_fn:
    :param fn_derivative_sign: 'pos' or 'neg':
    :param bracket_duration:
    :param n_brackets:
    :return: 2-tuple of Times bracketing the crossing, or None if no crossing found.
    """
    times = [time - i * bracket_duration for i in range(n_brackets + 1)]  # times[0] == time, reverse time.
    fn_values = metric_fn(times)
    if fn_derivative_sign == 'pos':
        has_crossing = [fn_values[i + 1] <= 0 and fn_values[i] > 0 for i in range(len(times) - 1)]
    elif fn_derivative_sign == 'neg':
        has_crossing = [fn_values[i + 1] > 0 and fn_values[i] <= 0 for i in range(len(times) - 1)]
    else:
        ValueError('fn_derivative_sign must be \'pos\' or \'neg\'.')
    if has_crossing.count(True) <= 0:
        return None, None, None
    i_found = has_crossing.index(True)
    return i_found, times, fn_values


def _refine_event_time(i_bracket, times, fn_values, metric_function):
    """ For a smooth metric function and initial timespan near event time, return a refined estimate
        (to given tolerance) of event time(i.e., at which metric function equals zero).
        Typically used in estimating target rise and set times (fn=altitude-target_horizon) and
        meridian transit times (fn=local hour angle).
    :param i_bracket: bracket index selected to refine. [int]
    :param times: list of times defining the brackets. [list of astropy.Time objects]
    :param fn_values: list of function values corresponding to times. [list of floats (dimensionless)]
    :param metric_function: a smooth function defined to be zero at time of desired event.
           This function takes datetime as sole parameter. [function object]
    :return: best estimate of event time. [py datetime]
    """
    def _refine_time(times, fn_values):
        if len(times) != 3 or len(fn_values) != 3:
            raise ValueError('_refine_time() requires exactly 3 equally spaced times and 3 fn values.')
        dt_hour = (times[1] - times[0]).sec / 3600.0  # time spacing in hours [float].
        t0, t1, t2 = tuple(times)
        y0, y1, y2 = tuple(fn_values)
        a = (y2 - 2 * y1 + y0) / (2 * dt_hour ** 2)
        b = (y2 - y0) / (2 * dt_hour)
        c = y1
        radical = sqrt((b * b) - (4 * a * c))
        root_minus = (-b - radical) / (2 * a)
        root_plus = (-b + radical) / (2 * a)
        time_root_minus = t1 + TimeDelta(root_minus * 3600, format='sec')
        time_root_plus = t1 + TimeDelta(root_plus * 3600, format='sec')
        time_linear_interp = t1 - TimeDelta((y1 / b) * 3600, format='sec')  # may depend on time spacing.
        distance_minus = abs(time_root_minus - time_linear_interp)
        distance_plus = abs(time_root_plus - time_linear_interp)
        if distance_minus < distance_plus:
            time_root = time_root_minus
        else:
            time_root = time_root_plus
        return time_root

    # Choose best adjacent third time to go with the two defined by i_bracket:
    if i_bracket == 0:
        i_trio = [0, 1, 2]
    elif i_bracket == len(times) - 2:
        i_trio = [len(times) - 3, len(times) - 2, len(times) - 1]
    elif abs(fn_values[i_bracket]) <= abs(fn_values[i_bracket + 1]):
        i_trio = [i_bracket - 1, i_bracket, i_bracket + 1]
    else:
        i_trio = [i_bracket, i_bracket + 1, i_bracket + 2]

    # Get interpolated time:
    times_3 = [times[i].copy() for i in i_trio]
    fn_values_3 = [fn_values[i].copy() for i in i_trio]
    time_interpolated = _refine_time(times_3, fn_values_3)

    # Correct the interpolated time, using new times and function values:
    dt_new = (times_3[1] - times_3[0]).sec / 3600.0 / 100.0  # new spacing, in hours.
    times_new = [time_interpolated + (i - 1) * TimeDelta(dt_new * 3600, format='sec') for i in range(3)]
    fn_values_new = metric_function(times_new)
    time_corrected = _refine_time(times_new, fn_values_new)
    return time_corrected


def workout(max_start_times=None, csv_filename='workout.csv'):
    """ Run both astropak.almanac.moon_rise_time() and astroplan.Observer.moon_rise_time()
        over a year, every approx. 30 minutes (dithered), in all 3 which modes.
        Save summary results in a .csv file, one line per (time, which) combination.
        Should be about 52,704 lines.
        Use Apache Point as location.
    """
    print('refreshing IERS_A.')
    astroplan.download_IERS_A()
    time_0 = Time('2021-01-01 00:00:00')
    d_seconds = 30 * 60  # 30 minutes.
    d_time = TimeDelta(d_seconds, format='sec')  # 30 minutes.
    if max_start_times is None:
        n_times = int(ceil(1 + (366 * 24 * 3600) / d_seconds))
    else:
        n_times = min(max_start_times, int(ceil(1 + (366 * 24 * 3600) / d_seconds)))
    dithering = 10 * 60  # seconds.
    seed(2022)
    print('Making all_start_times now.')
    all_start_times = [time_0 + i * d_time + TimeDelta(uniform(-dithering, +dithering), format='sec')
                       for i in range(n_times)]
    print(len(all_start_times), 'start times.')
    astroplan_site_name = 'APO'
    obs = astroplan.Observer.at_site(astroplan_site_name)
    print('Astroplan Site Name: ', astroplan_site_name)

    # Clear all setup for this expensive function, before starting to time while using it:
    _ = obs.moon_altaz(time_0).alt.deg
    slope_timedelta = TimeDelta(60, format='sec')

    workout_start_time = Time.now()
    output_header_line = ';'.join(['start_time',
                                   'which',
                                   'mrt_astropak',
                                   'ms',
                                   'err_us',
                                   'mrt_astroplan',
                                   'ms',
                                   'err_us'
                                   'diff']) + '\n'
    output_lines = [output_header_line]  # one .csv line per (start_time, which) combination.
    i_start_times_done = 0
    for start_time in all_start_times:
        for which in ('nearest', 'next', 'previous'):
            print('********** starting', '{0.iso}'.format(start_time), which)
            # First, handle astropak.almanac.moon_rise_time():
            ns_start = perf_counter_ns()
            mrt = moon_rise_time(obs, start_time, which)
            ns_end = perf_counter_ns()
            mrt_ms = round((ns_end - ns_start) / 1000000)
            if not mrt.masked:
                mrt_alt = obs.moon_altaz(mrt).alt.deg
                later_alt = obs.moon_altaz(mrt + slope_timedelta).alt.deg
                slope = (later_alt - mrt_alt) / slope_timedelta.sec  # deg/seconds
                mrt_error_usec = int(round(1000000 * mrt_alt / slope))  # microseconds
                mrt_error_used_string = str(mrt_error_usec)
            else:
                e = 'New moon_rise_time failed. If raised: '\
                    '(1) edit astropak to insert \'ERROR\', and '\
                    '(2) edit astroplan section below to calculate its own slope.'
                raise NotImplementedError(e)
            # Now, handle astroplan.Observer.moon_rise_time():
            ns_start = perf_counter_ns()
            astroplan_mrt = obs.moon_rise_time(start_time, which)
            ns_end = perf_counter_ns()
            astroplan_mrt_ms = round((ns_end - ns_start) / 1000000)
            if not astroplan_mrt.masked:
                # astroplan.Observer.moon_rise_time() succeeded.
                # Use slope from above.
                astroplan_mrt_alt = obs.moon_altaz(astroplan_mrt).alt.deg
                astroplan_mrt_error_usec = int(round(1000000 * astroplan_mrt_alt / slope))  # microseconds
                mrt_diff_msec = int(round(1000 * (mrt - astroplan_mrt).sec))
                astroplan_mrt_error_usec_string = str(astroplan_mrt_error_usec)
                mrt_diff_msec_string = str(mrt_diff_msec)
            else:
                # astroplan.Observer.moon_rise_time() failed.
                print('****************** obs.moon_rise_time() ERROR DETECTED. ******************')
                astroplan_mrt_error_usec_string = 'ERROR'
                mrt_diff_msec_string = 'ERROR'
            # Write line to output_data:
            # abs_mrt_diff_msec = abs(mrt_diff_msec)
            this_line = (';'.join(['{0.iso}'.format(start_time),
                                   which.ljust(8),
                                   '{0.iso}'.format(mrt),
                                   str(mrt_ms).rjust(4),
                                   mrt_error_used_string.rjust(8),
                                   '{0.iso}'.format(astroplan_mrt),
                                   str(astroplan_mrt_ms).rjust(4),
                                   astroplan_mrt_error_usec_string.rjust(8),
                                   mrt_diff_msec_string.rjust(5)])
                         + '\n')
            output_lines.append(this_line)
        i_start_times_done += 1
        if max_start_times is not None:
            if max_start_times >= 1:
                if i_start_times_done >= max_start_times:
                    break

    # Write output lines to new .csv file:
    fullpath = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, csv_filename)
    with open(fullpath, 'w') as f:
        f.writelines(output_lines)

    workout_end_time = Time.now()
    print('workout start time:', '{0.iso}'.format(workout_start_time))
    print('workout end time:  ', '{0.iso}'.format(workout_end_time))
    workout_hours = (workout_end_time - workout_start_time).sec / 3600
    print('workout required:', '{0:2f}'.format(workout_hours), 'hours.')
