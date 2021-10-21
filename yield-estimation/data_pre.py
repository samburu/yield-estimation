import datetime
import gzip
import tempfile
from pathlib import Path

import boto3
import botocore
import geopandas as gpd
import luigi
import pandas as pd
import rasterio
import requests
import xarray as xr
from kiluigi import IntermediateTarget
from luigi import ExternalTask, LocalTarget, Task
from luigi.util import requires
from rasterstats import zonal_stats

DATA_DIR = "/home/pmburu/data"

# No AWS keys required
CLIENT = boto3.client(
    "s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED)
)


class ScrapeERA5Var(ExternalTask):
    var = luigi.Parameter(default="air_temperature_at_2_metres")
    month = luigi.DateParameter(default=datetime.date(2012, 1, 1))

    def output(self):
        dst = Path(DATA_DIR) / self.var / f"{self.month.strftime('%Y%m')}_{self.var}.nc"
        return LocalTarget(path=str(dst))

    def run(self):
        year = self.month.strftime("%Y")
        month = self.month.strftime("%m")
        s3_data_key = f"{year}/{month}/data/{self.var}.nc"
        temp = tempfile.NamedTemporaryFile(suffix=".nc")
        era5_bucket = "era5-pds"
        CLIENT.download_file(era5_bucket, s3_data_key, temp.name)
        ds = xr.open_dataset(temp.name)
        ds = ds.mean(dim="time0")
        ds = ds.expand_dims({"month": [self.month.strftime("%Y-%m-%d")]})
        with self.output().open("w") as out:
            ds.to_netcdf(out.name)


class ScrapeRainfall(ExternalTask):
    month = luigi.DateParameter(default=datetime.date(2012, 1, 1))

    def output(self):
        dst = Path(DATA_DIR) / "chirps" / self.file_name()
        return LocalTarget(path=str(dst))

    def file_name(self):
        return f"chirps-v2.0.{self.month.strftime('%Y.%m')}.tif"

    def run(self):
        url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/EAC_monthly/tifs/{self.file_name()}.gz"
        req = requests.get(url)
        assert req.status_code == 200, req.raise_for_status()

        temp = tempfile.NamedTemporaryFile(suffix=".gz")
        with open(temp.name, "wb") as fd:
            for chunk in req.iter_content(chunk_size=1024):
                fd.write(chunk)

        with gzip.open(temp.name) as gzip_src:
            with rasterio.open(gzip_src) as src:
                arr = src.read()
                meta = src.meta.copy()
                meta.update(nodata=-9999.0)

        with self.output().open("w") as out:
            with rasterio.open(out.name, "w", **meta) as dst:
                dst.write(arr)


class ScrapeAdminLevel2(ExternalTask):
    def output(self):
        dst = Path(DATA_DIR) / "ken_adm_iebc_20191031_shp.zip"
        return LocalTarget(path=str(dst))

    def run(self):
        url = "https://data.kimetrica.com/dataset/dbe49118-c859-478a-9000-74c9487397b2/resource/ae58f714-b415-4f91-8072-4050fa577ee0/download/ken_admbnda_adm1_iebc_20191031.zip"

        req = requests.get(url)
        assert req.status_code == 200, req.raise_for_status()

        with self.output().open("w") as out:
            with open(out.name, "wb") as fd:
                for chunk in req.iter_content(chunk_size=1024):
                    fd.write(chunk)


@requires(ScrapeRainfall, ScrapeAdminLevel2)
class RainfallSummaryStats(Task):
    def output(self):
        return IntermediateTarget(task=self, timeout=1_036_800)

    def run(self):
        rain_src = self.input()[0].path
        admin_src = self.input()[1].path

        gdf = gpd.read_file(f"zip://{admin_src}")
        gdf = gdf[["ADM1_EN", "geometry"]]
        gdf = gdf.rename(columns={"ADM1_EN": "admin1"})
        stats = ["min", "max", "mean", "count", "median"]
        feature = zonal_stats(
            gdf, rain_src, stats=stats, all_touched=True, geojson_out=True
        )

        gdf = gpd.GeoDataFrame.from_features(feature)

        month_str = self.month.strftime("%b")
        gdf = gdf.rename(columns={i: f"{month_str}_rain_{i}" for i in stats})
        gdf = gdf.drop("geometry", 1)
        with self.output().open("w") as out:
            out.write(gdf)


class GetMaizeProduction(ExternalTask):
    def output(self):
        return IntermediateTarget(task=self, timeout=1_036_800)

    def run(self):
        df_12_16 = pd.read_csv(
            "http://kilimodata.org/dataset/fd95f913-608e-4881-96a3-d6fab8cc7fff/resource/847cccd3-5f2d-4583-bd26-e7ddb4a1657a/download/kenya-maize-production-by-counties-2012-2016.csv",
            encoding="latin1",
        )
        df_12_16.columns = df_12_16.loc[0].values
        df_12_16 = df_12_16.iloc[1:]
        df_12_16 = df_12_16.rename(
            columns={
                "COUNTY": "county",
                "Year": "year",
                "Harvested Area (HA)": "harvested_area",
                "Production (MT)": "production_mt",
                "Yield (MT/HA)": "yield_mt_ha",
            }
        )
        df_17 = pd.read_csv(
            "http://kilimodata.org/dataset/fd95f913-608e-4881-96a3-d6fab8cc7fff/resource/63fa57a5-a7c2-40e1-a728-f9546fa383ef/download/maize-production-and-value-by-counties-2017.csv",
            encoding="latin1",
        )
        df_17 = df_17.rename(
            columns={
                "County": "county",
                " Area (HA)  ": "harvested_area",
                " Production (MT)  ": "production_mt",
                " Value (Ksh)  ": "value_ksh",
            }
        )
        df_17["year"] = 2017
        df_18 = pd.read_csv(
            "http://kilimodata.org/dataset/fd95f913-608e-4881-96a3-d6fab8cc7fff/resource/e1a72ec2-e347-4264-8ac9-d4ec3659610a/download/maize-production-and-value-by-counties-2018.csv",
            encoding="latin1",
        )
        df_18 = df_18.rename(
            columns={
                "COUNTY": "county",
                "SUBCOUNTY": "subcounty",
                "Season": "season",
                "Area (Ha)": "harvested_area",
                "Quantity (Ton)": "production_mt",
                "Value (KShs)": "value_ksh",
            }
        )
        df_18["year"] = 2018
        df_18 = df_18.groupby(["county"])[
            "harvested_area", "production_mt", "value_ksh"
        ].sum()
        df_18 = df_18.reset_index()
        df = pd.concat([df_12_16, df_17, df_18])
        df["county"] = df["county"].str.title()

        standardize_county = {
            "Murang\x92A": "Murang'a",
            "Murang'A": "Murang'a",
            "MurangA": "Murang'a",
            "Nandi ": "Nandi",
            "Transnzoia": "Trans Nzoia",
            "Taita/Taveta": "Taita Taveta",
            "Tharaka Nithi": "Tharaka-Nithi",
            "Tharaka-Nthi": "Tharaka-Nithi",
            "Elgeyo/Marakwet": "Elgeyo-Marakwet",
            "Elgeyo Marakwet": "Elgeyo-Marakwet",
            "Nairobi ": "Nairobi",
        }
        df["county"] = df["county"].replace(standardize_county)
        with self.output().open("w") as out:
            out.write(df)


class SoilSandContent(ExternalTask):
    def output(self):
        return LocalTarget(
            "/home/pmburu/repo/yield-estimation/yield-estimation/data/af_SNDPPT_T__M_sd1_250m.tif"
        )


@requires(SoilSandContent, ScrapeAdminLevel2)
class SoilSandSummary(Task):
    def output(self):
        return IntermediateTarget(task=self, timeout=1_036_800)

    def run(self):
        rain_src = self.input()[0].path
        admin_src = self.input()[1].path

        gdf = gpd.read_file(f"zip://{admin_src}")
        gdf = gdf[["ADM1_EN", "geometry"]]
        gdf = gdf.rename(columns={"ADM1_EN": "admin1"})
        stats = ["min", "max", "mean", "count", "median"]
        feature = zonal_stats(
            gdf, rain_src, stats=stats, all_touched=True, geojson_out=True
        )
        gdf = gpd.GeoDataFrame.from_features(feature)
        gdf = gdf.rename(columns={i: f"sand_{i}" for i in stats})
        gdf = gdf.drop("geometry", 1)
        with self.output().open("w") as out:
            out.write(gdf)


class SoilClayContent(ExternalTask):
    def output(self):
        return LocalTarget(
            "/home/pmburu/repo/yield-estimation/yield-estimation/data/af_CLYPPT_T__M_sd1_250m.tif"
        )


@requires(SoilClayContent, ScrapeAdminLevel2)
class SoilClaySummary(Task):
    def output(self):
        return IntermediateTarget(task=self, timeout=1_036_800)

    def run(self):
        rain_src = self.input()[0].path
        admin_src = self.input()[1].path

        gdf = gpd.read_file(f"zip://{admin_src}")
        gdf = gdf[["ADM1_EN", "geometry"]]
        gdf = gdf.rename(columns={"ADM1_EN": "admin1"})
        stats = ["min", "max", "mean", "count", "median"]
        feature = zonal_stats(
            gdf, rain_src, stats=stats, all_touched=True, geojson_out=True
        )
        gdf = gpd.GeoDataFrame.from_features(feature)
        gdf = gdf.rename(columns={i: f"clay_{i}" for i in stats})
        gdf = gdf.drop("geometry", 1)
        with self.output().open("w") as out:
            out.write(gdf)


class SoilBulkDensity(ExternalTask):
    def output(self):
        return LocalTarget(
            "/home/pmburu/repo/yield-estimation/yield-estimation/data/af_BLD_T__M_sd1_250m.tif"
        )


@requires(SoilBulkDensity, ScrapeAdminLevel2)
class SoilBulkDensitySummary(Task):
    def output(self):
        return IntermediateTarget(task=self, timeout=1_036_800)

    def run(self):
        rain_src = self.input()[0].path
        admin_src = self.input()[1].path

        gdf = gpd.read_file(f"zip://{admin_src}")
        gdf = gdf[["ADM1_EN", "geometry"]]
        gdf = gdf.rename(columns={"ADM1_EN": "admin1"})
        stats = ["min", "max", "mean", "count", "median"]
        feature = zonal_stats(
            gdf, rain_src, stats=stats, all_touched=True, geojson_out=True
        )
        gdf = gpd.GeoDataFrame.from_features(feature)
        gdf = gdf.rename(columns={i: f"bulkdensity_{i}" for i in stats})
        gdf = gdf.drop("geometry", 1)
        with self.output().open("w") as out:
            out.write(gdf)


class SoilSiltContent(ExternalTask):
    def output(self):
        return LocalTarget(
            "/home/pmburu/repo/yield-estimation/yield-estimation/data/af_SLTPPT_T__M_sd1_250m.tif"
        )


@requires(SoilSiltContent, ScrapeAdminLevel2)
class SoilSiltSummary(Task):
    def output(self):
        return IntermediateTarget(task=self, timeout=1_036_800)

    def run(self):
        rain_src = self.input()[0].path
        admin_src = self.input()[1].path

        gdf = gpd.read_file(f"zip://{admin_src}")
        gdf = gdf[["ADM1_EN", "geometry"]]
        gdf = gdf.rename(columns={"ADM1_EN": "admin1"})
        stats = ["min", "max", "mean", "count", "median"]
        feature = zonal_stats(
            gdf, rain_src, stats=stats, all_touched=True, geojson_out=True
        )
        gdf = gpd.GeoDataFrame.from_features(feature)
        gdf = gdf.rename(columns={i: f"silt_{i}" for i in stats})
        gdf = gdf.drop("geometry", 1)
        with self.output().open("w") as out:
            out.write(gdf)


class TrainingData(Task):
    def requires(self):
        rain_task = self.clone(RainfallSummaryStats)
        month_list = pd.date_range(
            datetime.date(2012, 1, 1), datetime.date(2019, 1, 1), freq="M"
        )
        rain_task_map = {i: rain_task.clone(month=i) for i in month_list}
        return {
            "rain": rain_task_map,
            "yield": GetMaizeProduction(),
            "silt": SoilSiltSummary(),
            "bulk": SoilBulkDensitySummary(),
            "clay": SoilClaySummary(),
        }

    def output(self):
        return LocalTarget(
            path="/home/pmburu/repo/yield-estimation/yield-estimation/data/data.csv"
        )

    def run(self):
        input_map = self.input()
        year_map = {}
        for k, v in input_map["rain"].items():
            input_map["rain"][k] = (
                input_map["rain"][k].open().read().set_index("admin1")
            )
        years = set(i.year for i in input_map["rain"])
        for year in years:
            year_map[year] = pd.concat(
                [v for k, v in input_map["rain"].items() if k.year == year], 1
            )
        with input_map["silt"].open() as src:
            df_silt = src.read()
        with input_map["bulk"].open() as src:
            df_bulk = src.read()
        with input_map["clay"].open() as src:
            df_clay = src.read()
        # breakpoint()
        df = pd.concat(year_map)
        for _df in [df_silt, df_bulk, df_clay]:
            df = df.merge(_df, on="admin1", how="left")
        with input_map["yield"].open() as src:
            df_yield = src.read()
        df_yield = df_yield.drop(["Unnamed: 4", "Unnamed: 5"], 1)
        df_yield = df_yield.rename(columns={"county": "admin1"})
        df = df.T.drop_duplicates().T
        df = df.merge(df_yield, on=["year", "admin1"], how="outer")
        with self.output().open("w") as out:
            df.to_csv(out.name, index=False)


if __name__ == "__main__":
    month_list = pd.date_range(
        datetime.date(2012, 1, 1), datetime.date(2019, 1, 1), freq="M"
    )
    # luigi.build(
    #     [ScrapeERA5Var(month=i) for i in month_list], workers=10, local_scheduler=True
    # )
    luigi.build([TrainingData()], workers=1, local_scheduler=True)
