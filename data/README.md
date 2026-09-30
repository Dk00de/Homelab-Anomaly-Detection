# Datos

Esta carpeta no se versiona (ver `.gitignore`). Estructura esperada:

```
data/
├── raw/
│   ├── csic_database.csv   # CSIC 2010 HTTP Dataset (versión CSV de Kaggle)
│   └── npm_access.log      # access logs de Nginx Proxy Manager del homelab, anonimizados
└── processed/              # generado por `python -m hlad.build_features`
```

## CSIC 2010

Dataset público de peticiones HTTP a una tienda web de prueba, generado por el CSIC (Consejo Superior de Investigaciones Científicas, España). 36 000 peticiones normales y 25 065 anómalas (SQLi, XSS, path traversal, CRLF, buffer overflow, peticiones ilegales no intencionadas…). Columnas usadas: `Method`, `URL`, `content` (body de los POST), `classification` (0 normal / 1 anómalo).

## Logs del homelab

Access logs de Nginx Proxy Manager (`/opt/docker/nginx-proxy-manager/data/logs/proxy-host-*_access.log*`), del 2026-06-09 al 2026-09-29, unas 22 000 peticiones. Las IPs se reemplazaron por alias consistentes (`TS_n` Tailscale, `LAN_n` red privada, `PUB_n` pública) antes de salir del servidor:

```bash
anon() {
  perl -pe 'BEGIN{our %m; our $n=0}
    s{\b(\d{1,3}(?:\.\d{1,3}){3})\b}{
      my $ip=$1;
      my $k = $ip =~ /^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\./ ? "TS"
            : $ip =~ /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|127\.)/ ? "LAN" : "PUB";
      $m{$ip} //= $k."_".(++$n); $m{$ip}
    }ge;
    s/MAC=\S+/MAC=xx/g'
}
L=/opt/docker/nginx-proxy-manager/data/logs
sudo sh -c "zcat -f $L/proxy-host-*_access.log*" | anon > ~/npm_access.log
```

El servidor solo es accesible por Tailscale: en todo el periodo no hubo baneos de fail2ban ni bloqueos de UFW desde internet. Por eso **todo el tráfico real se considera clase normal** y los ataques de prueba son sintéticos.
