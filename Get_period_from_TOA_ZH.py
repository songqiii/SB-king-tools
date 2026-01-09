import numpy as np
from astropy import time, coordinates as coord, units as u
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
from astropy.utils.data import clear_download_cache
from astropy.utils.iers import IERS_Auto
import argparse
import sys

def update_iers_cache():
    """更新IERS数据缓存"""
    clear_download_cache()
    IERS_Auto.iers_table = IERS_Auto.open()
    
    # 可选：配置自动下载
    from astropy.utils.iers import conf
    conf.auto_max_age = None

def get_error_new(x, numbers):
    """计算误差函数"""
    return np.mean(np.abs((numbers / x) - np.round(numbers / x)))

def find_best_period(numbers, step=1e-5, search_range=(0.1, 10.0)):
    """
    在指定范围内搜索最佳周期
    
    Parameters:
    -----------
    numbers : array_like
        时间间隔数组
    step : float
        搜索步长
    search_range : tuple
        搜索范围 (min, max)
    
    Returns:
    --------
    best_period : float
        最佳周期值
    min_error : float
        最小误差值
    """
    periods = np.arange(search_range[0], search_range[1], step)
    errors = []
    
    for period in periods:
        errors.append(get_error_new(period, numbers))
    
    errors = np.array(errors)
    min_error_index = np.argmin(errors)
    min_error = errors[min_error_index]
    best_period = periods[min_error_index]
    
    return best_period, min_error

def read_times_from_file(filename):
    """
    从文本文件读取时间数据
    
    文件格式：每行一个MJD时间（UTC）
    示例：
    60982.24344730433
    60982.24523075026
    60982.245293727115
    """
    times = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释
                try:
                    times.append(float(line))
                except ValueError:
                    print(f"警告: 无法解析行: {line}")
    return np.array(times)

def utc_to_barycentric(utc_times, location, source):
    """
    将UTC时间转换为质心动力学时
    
    Parameters:
    -----------
    utc_times : array_like
        UTC时间数组（MJD格式）
    location : EarthLocation
        观测站位置
    source : SkyCoord
        源的天球坐标
    
    Returns:
    --------
    barycentric_times : array
        质心动力学时数组（秒）
    """
    bary_times = []
    
    for utc_mjd in utc_times:
        t_utc = time.Time(utc_mjd, format='mjd', scale='utc', location=location)
        ltt_bary = t_utc.light_travel_time(source)
        t_bary = t_utc.tdb + ltt_bary
        bary_times.append(t_bary.value)
    
    bary_times = np.array(bary_times) * 24 * 3600  # 转换为秒
    return bary_times

def get_source_coordinates(source_name):
    """
    根据源名称或坐标获取天球坐标
    
    Parameters:
    -----------
    source_name : str
        源名称 ('ASKAP', '1905-beam4', '1905-dejiang') 或坐标字符串
    
    Returns:
    --------
    source_coord : SkyCoord
        源的天球坐标
    """
    # 已知源的坐标映射
    known_sources = {
        'ASKAP': ("16:06:53", "-08:54:07"),
        '1905-beam4': ("19:05:29.22", "-01:28:16.4"),
        '1905-dejiang': ("19:05:33.20", "-01:28:27.0"),
    }
    
    if source_name in known_sources:
        ra_str, dec_str = known_sources[source_name]
        print(f"使用已知源: {source_name}, 坐标: RA={ra_str}, Dec={dec_str}")
        return coord.SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame='icrs')
    else:
        # 如果输入了未知的源名称，尝试将其作为坐标字符串
        try:
            # 尝试直接解析输入的字符串作为坐标
            # 格式可以是 "16:06:53 -08:54:07" 或 "16:06:53, -08:54:07"
            if "," in source_name:
                ra_str, dec_str = [s.strip() for s in source_name.split(",")]
            else:
                # 假设用空格分隔
                parts = source_name.strip().split()
                if len(parts) == 2:
                    ra_str, dec_str = parts
                else:
                    # 尝试将多个空格的部分合并
                    ra_str = parts[0]
                    dec_str = " ".join(parts[1:])
            
            print(f"使用自定义坐标: RA={ra_str}, Dec={dec_str}")
            return coord.SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame='icrs')
        except Exception as e:
            raise ValueError(f"无法解析源坐标 '{source_name}': {e}。请使用已知源名称或提供有效的坐标字符串（如 '16:06:53 -08:54:07'）")

def get_observatory_location(obs_name):
    """
    根据观测站名称获取观测站位置
    
    Parameters:
    -----------
    obs_name : str
        观测站名称 ('FAST', 'Parkes', 'CHIME')
    
    Returns:
    --------
    location : EarthLocation
        观测站位置
    """
    observatories = {
        'FAST': EarthLocation(lon='106.857833 deg', lat='25.652557 deg', height='900 m'),
        'Parkes': EarthLocation(lon='148.26409167643106 deg', lat='-32.997842087902235 deg', height='400 m'),
        'CHIME': EarthLocation(lon='119.623611 deg', lat='49.3208333 deg', height='545 m')
    }
    
    if obs_name in observatories:
        return observatories[obs_name]
    else:
        raise ValueError(f"未知的观测站: {obs_name}。请使用以下之一: {', '.join(observatories.keys())}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='从脉冲星爆发时间数据中搜索最佳周期',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python pulsar_period_search.py times.txt --step 1e-5
  python pulsar_period_search.py times.txt --step 1e-6 --min 0.01 --max 5.0
  python pulsar_period_search.py times.txt --source 1905-beam4
  python pulsar_period_search.py times.txt --location CHIME
  python pulsar_period_search.py times.txt --source "16:06:53 -08:54:07"
  python pulsar_period_search.py times.txt --source "19:05:29.22 -01:28:16.4"
  python pulsar_period_search.py times.txt --output results.txt
        """)
    
    parser.add_argument('input_file', type=str, 
                       help='输入文件路径，包含MJD UTC时间（每行一个）')
    parser.add_argument('--step', type=float, default=1e-5,
                       help='搜索步长（默认: 1e-5）')
    parser.add_argument('--min', type=float, default=0.1,
                       help='最小搜索周期（秒，默认: 0.1）')
    parser.add_argument('--max', type=float, default=10.0,
                       help='最大搜索周期（秒，默认: 10.0）')
    parser.add_argument('--location', type=str, default='FAST',
                       choices=['FAST', 'Parkes', 'CHIME'],
                       help='观测站位置（默认: FAST）')
    parser.add_argument('--source', type=str, default='ASKAP',
                       help='源名称: ASKAP, 1905-beam4, 1905-dejiang 或坐标字符串（如 "16:06:53 -08:54:07"）（默认: ASKAP）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（如果指定，将结果保存到文件）')
    parser.add_argument('--update_iers', action='store_true',
                       help='更新IERS数据缓存')
    
    args = parser.parse_args()
    
    # 更新IERS数据（如果指定）
    if args.update_iers:
        print("正在更新IERS数据...")
        update_iers_cache()
    
    # 读取时间数据
    print(f"正在读取文件: {args.input_file}")
    try:
        utc_times = read_times_from_file(args.input_file)
        print(f"读取到 {len(utc_times)} 个时间点")
    except FileNotFoundError:
        print(f"错误: 文件 '{args.input_file}' 未找到")
        sys.exit(1)
    
    if len(utc_times) < 2:
        print("错误: 至少需要2个时间点")
        sys.exit(1)
    
    # 设置观测站位置
    try:
        observatory = get_observatory_location(args.location)
        print(f"使用{args.location}观测站位置")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 设置源坐标
    try:
        source_coord = get_source_coordinates(args.source)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 转换为质心动力学时
    print("正在转换为质心动力学时...")
    bary_times = utc_to_barycentric(utc_times, observatory, source_coord)
    
    # 计算时间间隔
    intervals = np.diff(bary_times)
    
    print("\n时间间隔（秒）:")
    np.set_printoptions(suppress=True, precision=6)
    print(intervals)
    
    # 搜索最佳周期
    print(f"\n正在搜索最佳周期（范围: {args.min} - {args.max} 秒，步长: {args.step})...")
    best_period, min_error = find_best_period(
        intervals, 
        step=args.step, 
        search_range=(args.min, args.max)
    )
    
    print(f"\n结果:")
    print(f"最佳周期: {best_period:.10f} 秒")
    print(f"最小误差: {min_error:.10f}")
    
    # 计算并显示拟合结果
    ratios = intervals / best_period
    rounded_ratios = np.round(ratios)
    residuals = ratios - rounded_ratios
    
    print(f"\n间隔/周期比值（整数部分表示周期数）:")
    for i, (interval, ratio, rounded, residual) in enumerate(zip(
        intervals, ratios, rounded_ratios, residuals)):
        print(f"间隔 {i+1}: {interval:.6f} s = {ratio:.6f} 周期 "
              f"(最近整数: {int(rounded)}, 残差: {residual:.6f})")
    
    print(f"\n残差统计:")
    print(f"  平均值: {np.mean(residuals):.6f}")
    print(f"  标准差: {np.std(residuals):.6f}")
    print(f"  最大绝对值: {np.max(np.abs(residuals)):.6f}")
    
    # 只有当指定了输出文件时才写入文件
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# 周期搜索结果\n")
            f.write(f"# 输入文件: {args.input_file}\n")
            f.write(f"# 观测站: {args.location}\n")
            f.write(f"# 源: {args.source}\n")
            f.write(f"# 搜索范围: {args.min} - {args.max} 秒\n")
            f.write(f"# 搜索步长: {args.step}\n\n")
            f.write(f"最佳周期: {best_period:.10f} 秒\n")
            f.write(f"最小误差: {min_error:.10f}\n\n")
            f.write("序号,间隔(秒),周期数,最近整数周期,残差\n")
            for i, (interval, ratio, rounded, residual) in enumerate(zip(
                intervals, ratios, rounded_ratios, residuals)):
                f.write(f"{i+1},{interval:.6f},{ratio:.6f},{int(rounded)},{residual:.6f}\n")
        
        print(f"\n详细结果已保存到: {args.output}")
    else:
        print("\n注意: 未指定输出文件，结果仅显示在屏幕上。使用 --output 参数可将结果保存到文件。")

if __name__ == "__main__":
    main()