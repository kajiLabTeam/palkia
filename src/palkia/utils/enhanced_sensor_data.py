from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from palkia.const.column_name import (
    ACC_X,
    ACC_Y,
    ACC_Z,
    GYRO_X,
    GYRO_Y,
    GYRO_Z,
    TIMESTAMP,
)


class EnhancedSensorData:
    def __init__(self, acc_df: pd.DataFrame, gyro_df: pd.DataFrame) -> None:
        self.acc_df = self._validate_and_process_acc(acc_df)
        self.gyro_df = self._validate_and_process_gyro(gyro_df)
        self._sync_timestamps()

    @staticmethod
    def _validate_and_process_acc(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [TIMESTAMP, ACC_X, ACC_Y, ACC_Z]
        if not all(col in df.columns for col in required_columns):
            msg = f"Acceleration dataframe must contain columns: {required_columns}"
            raise ValueError(msg)
        return df[required_columns].sort_values(TIMESTAMP).reset_index(drop=True)

    @staticmethod
    def _validate_and_process_gyro(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [TIMESTAMP, GYRO_X, GYRO_Y, GYRO_Z]
        if not all(col in df.columns for col in required_columns):
            msg = f"Gyroscope dataframe must contain columns: {required_columns}"
            raise ValueError(msg)
        return df[required_columns].sort_values(TIMESTAMP).reset_index(drop=True)

    def _sync_timestamps(self) -> None:
        # Ensure that acc_df and gyro_df have the same timestamps
        acc_timestamps = pd.Index(self.acc_df[TIMESTAMP])
        gyro_timestamps = pd.Index(self.gyro_df[TIMESTAMP])
        common_timestamps = acc_timestamps.intersection(gyro_timestamps)
        self.acc_df = self.acc_df[self.acc_df[TIMESTAMP].isin(common_timestamps)]
        self.gyro_df = self.gyro_df[self.gyro_df[TIMESTAMP].isin(common_timestamps)]

    def get_acc_data(self) -> pd.DataFrame:
        return self.acc_df

    def get_gyro_data(self) -> pd.DataFrame:
        return self.gyro_df

    def get_merged_data(self) -> pd.DataFrame:
        return pd.merge(self.acc_df, self.gyro_df, on=TIMESTAMP)

    def resample(self, freq: str) -> EnhancedSensorData:
        resampled_acc = (
            self.acc_df.set_index(TIMESTAMP).resample(freq).mean().reset_index()
        )
        resampled_gyro = (
            self.gyro_df.set_index(TIMESTAMP).resample(freq).mean().reset_index()
        )
        return EnhancedSensorData(resampled_acc, resampled_gyro)

    def filter_data(
        self, start_time: float | None = None, end_time: float | None = None
    ) -> EnhancedSensorData:
        mask_acc = self._create_time_mask(self.acc_df[TIMESTAMP], start_time, end_time)
        mask_gyro = self._create_time_mask(
            self.gyro_df[TIMESTAMP], start_time, end_time
        )
        return EnhancedSensorData(self.acc_df[mask_acc], self.gyro_df[mask_gyro])

    @staticmethod
    def _create_time_mask(
        timestamps: pd.Series, start_time: float | None, end_time: float | None
    ) -> pd.Series:
        mask = pd.Series(True, index=timestamps.index)
        if start_time is not None:
            mask &= timestamps >= start_time
        if end_time is not None:
            mask &= timestamps <= end_time
        return mask

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.acc_df.to_numpy(), self.gyro_df.to_numpy()

    def __len__(self) -> int:
        return len(self.acc_df)

    def __str__(self) -> str:
        return f"SensorData(acc_samples={len(self.acc_df)}, gyro_samples={len(self.gyro_df)})"

    def __repr__(self) -> str:
        return self.__str__()
