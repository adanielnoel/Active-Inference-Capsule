from typing import Union, Any, List
from collections.abc import Iterable
from collections import defaultdict


"""
Utility class for logging time-series data and doing some handy operations with it
"""


class Timeline:
    def __init__(self, time_max_decimals=5):
        self.tml = defaultdict(dict)
        self.time_max_decimals = time_max_decimals

    @property
    def times(self):
        return sorted(list(self.tml.keys()))

    def get_frame(self, time):
        return self.tml[time]

    def log(self, time: Union[int, float, List[Union[int, float]], Any], key: str, value: Union[Any, List[Any]]):
        """
        Add an entry for a particular time-step. Raises exception if the key already exists for that time-step.
        It is also possible to add entries in batch by providing a list of times and corresponding values.
        """
        if issubclass(type(time), Iterable):
            if issubclass(type(value), Iterable):
                assert len(time) == len(value), "Length of arrays <time> and <value> does not match"
            else:
                value = [value for _ in time]
            for i in range(len(time)):
                self.log(float(time[i]), key, value[i])
        else:
            time = round(float(time), self.time_max_decimals)
            if key in self.tml[time].keys():
                raise KeyError(f'key <{key}> already exists at t={time}')
            else:
                self.tml[time][key] = value

    def select_features(self, keys: Union[str, List[str]]):
        """
        Returns all times and values for the given keys
        """
        keys = [keys] if isinstance(keys, str) else keys
        times = []
        values = [[] for _ in keys]
        for time in sorted(self.times):  # make sure the return time-series is sorted (increasing time)
            if not all([key in self.tml[time].keys() for key in keys]):
                continue    # Only include time-step if all keys have a datapoint in it
            times.append(time)
            for i, key in enumerate(keys):
                values[i].append(self.tml[time][key])
        if len(values) == 1:
            return times, values[0]
        else:
            return times, values

    def delete_feature(self, key):
        for time in self.times:
            if key in list(self.tml[time].keys()):
                del self.tml[time][key]

    def merge(self, other: 'Timeline'):
        new_times = sorted(list(set(self.times + other.times)))
        self_dt = self.times[-1] - self.times[-2]
        other_dt = other.times[-1] - other.times[-2]
        new_dt = new_times[-1] - new_times[-2]
        self_resampled = self.resample(new_times, extend_steps_ends=int(round(self_dt / new_dt, self.time_max_decimals)) - 1)
        other_resampled = other.resample(new_times, extend_steps_ends=int(round(other_dt / new_dt, self.time_max_decimals)) - 1)
        merged = Timeline()
        for t in new_times:
            for key, value in self_resampled.tml[t].items():
                merged.log(t, key, value)
            for key, value in other_resampled.tml[t].items():
                merged.log(t, key, value)
        return merged

    def resample(self, new_times, extend_steps_ends=0, keep_outside_times=True):
        """
        Returns a new Timeline object sampled at the new times.
        The values are selected from the closest available logged time-step.
        If the logs contain Timeline objects as values, these will be expanded recursively.
        If keep_outside_times==True (default), the times outside the range of new_times will be kept.
        """
        new_tml = Timeline()
        new_delta_t = 0 if not extend_steps_ends else new_times[-1] - new_times[-2]
        if extend_steps_ends:
            new_times = list(new_times) + [new_times[-1] + i * new_delta_t for i in range(1, extend_steps_ends + 1)]
        # Resample non-timeline objects
        resampled_times = []
        for new_t in new_times:
            try:
                closest_t = self.times[max([i for i, t in enumerate(self.times) if t <= new_t])]  # Largest available time lower than new_t
                resampled_times.append(closest_t)
            except ValueError:
                continue  # No time-step found before new_t, skip
            for key, value in self.tml[closest_t].items():
                if not isinstance(value, Timeline):
                    new_tml.log(new_t, key, value)
        # Include times outside the range of new_times
        if keep_outside_times:
            outside_times = list(set(self.times) - set(resampled_times))
            for time in outside_times:
                for key, value in self.tml[time].items():
                    if not isinstance(value, Timeline):
                        new_tml.log(time, key, value)
        # Recursively resample timeline objects
        for current_t in self.times:
            for key, value in self.tml[current_t].items():
                if isinstance(value, Timeline):
                    start_t = min(value.times)
                    end_t = max(value.times) + extend_steps_ends * new_delta_t
                    resampled = value.resample([t for t in new_times if start_t <= t <= end_t])
                    closest_t = new_times[min([i for i, t in enumerate(new_times) if t >= current_t])]  # Smallest new time larger than current_t
                    new_tml.log(closest_t, key, resampled)
        return new_tml


if __name__ == '__main__':
    import torch
    import numpy as np

    tml = Timeline()
    tml.log(np.arange(0.0, 0.4, 0.1), 'torch', torch.randn(4))
    tml.log(0.1, 'val', 0.1)
    tml.log(0.2, 'val', 0.4)
    tml_sub = Timeline()
    tml_sub.log([0.2, 0.3], 'val2', [0.1, 0.5])
    tml.log(0.2, 'sub', tml_sub)
    tml.log(0.3, 'val', 0.2)

    tml_other = Timeline()
    tml_other.log(np.arange(0.0, 0.4, 0.025), 'torch2', torch.randn(16))
    tml_merged = tml.merge(tml_other)

    tml2 = tml.resample(np.arange(0.0, 0.3, 0.05), extend_steps_ends=2)
    times_, values_ = tml2.select_feature('sub')
    print(times_)
    print(values_)
