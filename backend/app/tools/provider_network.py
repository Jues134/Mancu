"""Keep LAN model traffic off desktop system proxies."""
from ipaddress import ip_address, ip_network
from urllib.parse import urlsplit

_LAN = tuple(ip_network(value) for value in (
    '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '127.0.0.0/8',
    '169.254.0.0/16', '::1/128', 'fc00::/7', 'fe80::/10',
))

def provider_trust_env(url: str) -> bool:
    """Public providers retain proxy configuration; explicit LAN hosts go direct."""
    host = (urlsplit(url).hostname or '').lower().rstrip('.')
    if host == 'localhost' or host.endswith(('.localhost', '.local')):
        return False
    try:
        address = ip_address(host)
    except ValueError:
        return True
    if address.version == 6 and address.ipv4_mapped:
        address = address.ipv4_mapped
    return not any(address in network for network in _LAN)
