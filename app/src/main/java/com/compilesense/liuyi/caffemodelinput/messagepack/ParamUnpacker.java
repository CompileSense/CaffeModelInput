package com.compilesense.liuyi.caffemodelinput.messagepack;

import android.util.Log;

import org.msgpack.MessagePack;
import org.msgpack.unpacker.Unpacker;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;

/**
 * 参数读取
 *
 * model 中的数据用 msgpack 序列化,在 java 层读取。
 * Created by shenjingyuan002 on 2017/3/3.
 */

public class ParamUnpacker {
    public Object[] unpackFunction(String paramFilePath, Class[] classTypes){
        Object[] objects = new Object[2];
        try {
            Unpacker unpacker = read_parameters_MessagePack_Init(paramFilePath);
            objects[0] = unpacker.read(classTypes[0]);
            objects[1] = unpacker.read(classTypes[1]);
            unpacker.close();
            unpacker = null;
        } catch (Exception e) {
            Log.d("MessagePack Exception:", e.getMessage());
            return null;
        }
        return objects;
    }

    public Object unpackFunction(String paramFilePath, Class classType) {
        Object convParams;
        try {
            Unpacker unpacker = read_parameters_MessagePack_Init(paramFilePath);
            convParams = unpacker.read(classType);
            unpacker.close();
        } catch (Exception e) {
            Log.d("MessagePack_Init:", e.getMessage());
            return null;
        }

        return convParams;
    }

    private Unpacker read_parameters_MessagePack_Init(String name) {
        MessagePack msgpack = new MessagePack();
        File f = new File(name);
        byte[] nbytes = null;

        try {
            FileInputStream fin = new FileInputStream(f);
            nbytes = new byte[(int)f.length()];
            fin.read(nbytes);
            fin.close();

        } catch (Exception e) {
            Log.d("MessagePack_Init:", e.getMessage());
            return null;
        }

        ByteArrayInputStream in = new ByteArrayInputStream(nbytes);
        Unpacker unpack = msgpack.createUnpacker(in);
        return unpack;
    }
}
