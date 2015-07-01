#ifndef TCPINSTRUMENT_H
#define TCPINSTRUMENT_H

#include <QTcpSocket>

#include "communicationprotocol.h"

class TcpInstrument : public CommunicationProtocol
{
    Q_OBJECT
public:
    explicit TcpInstrument(QString key, QString subKey, QObject *parent = nullptr);
    ~TcpInstrument();

    bool writeCmd(QString cmd);
    bool writeBinary(QByteArray dat);
    QByteArray queryCmd(QString cmd);
    QIODevice *device(){ return p_socket; }

public slots:
    virtual void initialize();
    virtual bool testConnection();
    void socketError(QAbstractSocket::SocketError);


private:
    QString d_ip;
    int d_port;
    QTcpSocket *p_socket;

    bool connectSocket();
    void disconnectSocket();
    void setSocketConnectionInfo(QString ip, int port);

};

#endif // TCPINSTRUMENT_H
