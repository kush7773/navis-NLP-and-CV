#!/bin/bash
# Generate self-signed certificate for Navis Robot

echo "🔐 Generating Self-Signed Certificate..."

# Generate private key and certificate in one step
openssl req -new -newkey rsa:2048 -days 365 -nodes -x509 \
    -keyout key.pem -out cert.pem \
    -subj "/C=US/ST=State/L=City/O=Navis/CN=raspberrypi"

echo "✅ Certificate generated:"
echo "   - cert.pem (Public Certificate)"
echo "   - key.pem (Private Key)"
echo ""
echo "📝 Instructions:"
echo "1. Run the server: python3 voice_follow_integration.py"
echo "2. Access via HTTPS: https://<RASPBERRY_PI_IP>:5000"
echo "3. Accept the security warning in your browser"
