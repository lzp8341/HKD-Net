import os
import glob
import binascii
import numpy as np
from PIL import Image
import scapy.all as scapy
from tqdm import tqdm
import random

def makedir(path):
    """创建目录，忽略已存在的错误"""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Directory creation failed: {e}")

def preprocess_packet(packet):
    """
    预处理数据包：移除以太网头，匿名化IP和端口
    :param packet: Scapy解析的包对象
    :return: 预处理后的IP层包
    """
    if 'IP' not in packet:
        return None
        
    # 移除以太网头
    if 'Ether' in packet:
        packet = packet['IP']
    
    # 匿名化处理
    if 'TCP' in packet:
        packet['TCP'].sport = 0
        packet['TCP'].dport = 0
    elif 'UDP' in packet:
        packet['UDP'].sport = 0
        packet['UDP'].dport = 0
    
    # 替换IP地址但保持方向
    src_ip = packet.src
    dst_ip = packet.dst
    packet.src = "192.168." + ".".join([str(random.randint(1, 254)) for _ in range(2)])
    packet.dst = "10.0." + ".".join([str(random.randint(1, 254)) for _ in range(2)])
    
    return packet

def extract_packet_data(packet):
    """
    提取单个包的包头和负载，并标准化长度
    :param packet: 预处理后的Scapy包对象
    :return: (header_hex, payload_hex) 标准化后的十六进制字符串
    """
    if packet is None:
        return ('0'*160, '0'*480)
    
    header = binascii.hexlify(bytes(packet)).decode()
    payload = binascii.hexlify(bytes(packet['Raw'])).decode() if 'Raw' in packet else ''
    header = header[:160].ljust(160, '0')  # 不足补零，超长截断
    payload = payload[:480].ljust(480, '0')
    
    return header, payload

def build_mfr_matrix(packets, num_packets=5):
    """
    构建多级流表示（MFR）矩阵
    :param packets: 包列表（预处理后的Scapy对象）
    :param num_packets: 每个流包含的包数（默认为5）
    :return: 40x40的MFR矩阵（numpy数组）
    """
    mfr_matrix = np.zeros((40, 40), dtype=np.uint8)
    
    processed_packets = []
    for p in packets:
        pp = preprocess_packet(p)
        if pp is not None:
            processed_packets.append(pp)
            if len(processed_packets) >= num_packets:
                break
    
    for i in range(min(num_packets, len(processed_packets))):
        header, payload = extract_packet_data(processed_packets[i])
        header_bytes = [int(header[j:j+2], 16) for j in range(0, 160, 2)]
        mfr_matrix[i*8 : i*8+2, :] = np.array(header_bytes).reshape(2, 40)
        payload_bytes = [int(payload[j:j+2], 16) for j in range(0, 480, 2)]
        mfr_matrix[i*8+2 : i*8+8, :] = np.array(payload_bytes).reshape(6, 40)
    return mfr_matrix

def save_as_grayscale_image(matrix, output_path):
    """
    将MFR矩阵保存为灰度图像
    :param matrix: 40x40的MFR矩阵
    :param output_path: 输出图像路径
    """
    img = Image.fromarray(matrix, 'L')  
    img.save(output_path)

def process_pcap_to_mfr(pcap_path):
    """
    处理单个PCAP文件并生成MFR矩阵
    :param pcap_path: 输入PCAP文件路径
    :return: 40x40的MFR矩阵（numpy数组）
    """
    try:
        packets = scapy.rdpcap(pcap_path)
        mfr_matrix = build_mfr_matrix(packets)
        return mfr_matrix
    except Exception as e:
        print(f"Error processing {pcap_path}: {e}")
        return None

def generate_mfr_dataset(input_dir, output_dir):
    """
    批量处理PCAP文件生成MFR数据集，并保存为RGB图像
    :param input_dir: 包含PCAP文件的输入目录（按类别子目录组织）
    :param output_dir: 输出目录（保持相同结构）
    """
    makedir(output_dir)                                                           
    class_dirs = glob.glob(os.path.join(input_dir, '*'))
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = os.path.basename(class_dir)
        class_output_dir = os.path.join(output_dir, class_name)
        makedir(class_output_dir)
        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))                 
        for pcap_file in pcap_files:
            mfr_matrix = process_pcap_to_mfr(pcap_file)
            if mfr_matrix is not None:
                base_name = os.path.splitext(os.path.basename(pcap_file))[0]
                output_path = os.path.join(class_output_dir, f"{base_name}.png")
                save_as_grayscale_image(mfr_matrix, output_path)                  

if __name__ == "__main__":
    input_pcap_dir = "/pcap"  # 替换为实际路径
    output_mfr_dir = "/MRF_image"
    generate_mfr_dataset(input_pcap_dir, output_mfr_dir)

