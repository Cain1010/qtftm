#include "tcpinstrument.h"
#include <QTime>

TcpInstrument::TcpInstrument(QString key, QString name, QObject *parent) :
    HardwareObject(key,name,parent)
{
}

TcpInstrument::~TcpInstrument()
{
    disconnectSocket();
}

void TcpInstrument::initialize()
{
	d_socket = new QTcpSocket(this);

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	QString ip = s.value(key().append(QString("/ip")),QString("")).toString();
	int port = s.value(key().append(QString("/port")),5000).toInt();

	s.setValue(key().append(QString("/prettyName")),name());

	setSocketConnectionInfo(ip,port);
}

bool TcpInstrument::testConnection()
{
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	QString ip = s.value(key().append(QString("/ip")),QString("")).toString();
	int port = s.value(key().append(QString("/port")),5000).toInt();

	if(ip == d_ip && port == d_port && d_socket->state() == QTcpSocket::ConnectedState)
		return true;

	if(d_socket->state() != QTcpSocket::UnconnectedState)
		disconnectSocket();

    setSocketConnectionInfo(ip,port);

	return connectSocket();

}

void TcpInstrument::socketError(QAbstractSocket::SocketError)
{
	//consider handling errors here at the socket level
}

bool TcpInstrument::writeCmd(QString cmd)
{
    if(d_socket->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write command to %1. Socket is not connected. (Command = %2)").arg(d_prettyName).arg(cmd));
            return false;
        }
    }

    d_socket->write(cmd.toLatin1());
    if(!d_socket->flush())

    //int return_value =  d_socket->write(cmd.toLatin1());    // praa Nov 20 2014, see rs232instrument.cpp
    //d_socket->flush();
    //if(return_value == -1)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write command to %1. (Command = %2)").arg(d_prettyName).arg(cmd),QtFTM::LogError);
        return false;
    }
    return true;
}

QByteArray TcpInstrument::queryCmd(QString cmd)
{
    if(d_socket->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write query to %1. Socket is not connected. (Query = %2)").arg(d_prettyName).arg(cmd));
            return QByteArray();
        }
    }

    if(d_socket->bytesAvailable())
        d_socket->readAll();
    d_socket->write(cmd.toLatin1());
    if(!d_socket->flush())
    //int return_value =  d_socket->write(cmd.toLatin1());    // praa Nov 20 2014, see rs232instrument.cpp
    //d_socket->flush();
    //if(return_value == -1)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write query to %1. (query = %2)").arg(d_prettyName).arg(cmd),QtFTM::LogError);
        return QByteArray();
    }

	//write to socket here, return response
    if(!d_useTermChar || d_readTerminator.isEmpty())
    {
        if(!d_socket->waitForReadyRead(d_timeOut))
        {
            emit hardwareFailure();
            emit logMessage(QString("%1 did not respond to query. (query = %2)").arg(d_prettyName).arg(cmd),QtFTM::LogError);
            return QByteArray();
        }

        return d_socket->readAll();
    }
    else
    {
        QByteArray out;
        bool done = false;
        while(!done)
        {
            if(!d_socket->waitForReadyRead(d_timeOut))
                break;

            out.append(d_socket->readAll());
            if(out.endsWith(d_readTerminator))
                return out;
        }

        emit hardwareFailure();
        emit logMessage(QString("%1 timed out while waiting for termination character. (query = %2, partial response = %3)").arg(d_prettyName).arg(cmd).arg(QString(out)),QtFTM::LogError);
        emit logMessage(QString("Hex response: %1").arg(QString(out.toHex())));
        return out;
    }
    return QByteArray();
}

bool TcpInstrument::connectSocket()
{
    d_socket->connectToHost(d_ip,d_port);
    if(!d_socket->waitForConnected(1000))
    {
        emit logMessage(QString("Could not connect to %1 at %2:%3. %4").arg(d_prettyName).arg(d_ip).arg(d_port).arg(d_socket->errorString()),QtFTM::LogError);
        return false;
    }
    d_socket->setSocketOption(QAbstractSocket::KeepAliveOption,1);
    d_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);
    return true;
}

void TcpInstrument::disconnectSocket()
{
    d_socket->disconnectFromHost();
}

void TcpInstrument::setSocketConnectionInfo(QString ip, int port)
{
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	s.setValue(key().append(QString("/ip")),ip);
	s.setValue(key().append(QString("/port")),port);
	s.sync();

	d_ip = ip;
	d_port = port;
}
