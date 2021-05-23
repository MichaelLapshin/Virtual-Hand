using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Net.Sockets;
using Debug = UnityEngine.Debug;
using System.Windows.Input;

public class GeneralData : MonoBehaviour
{
    private static bool ready = false;
    private static List<HingeJoint> sortedAllHingeJoints;

    // Start is called before the first frame update
    void Start()
    {
    }

    public static bool isReady()
    {
        if (ready)
        {
            return true;
        }

        // Object fetching related logic
        HingeJoint[] allHingeObjects = FindObjectsOfType(
            typeof(HingeJoint)) as HingeJoint[]; // Should return all the finger limbs (since they have joints)
        sortedAllHingeJoints = allHingeObjects.OrderBy(go => go.name).ToList();

        ready = true;
        return ready;
    }

    public static UnityEngine.GameObject[] getHingeLimbs(string rootName)
    {
        // Sifts through the hinged objects to determine the ones we want
        var sortedHingeJoints = new ArrayList();
        foreach (var hinge in sortedAllHingeJoints)
        {
            if (GeneralData.isAncestor(hinge.transform, rootName) == true)
            {
                sortedHingeJoints.Add(hinge);
            }
        }

        // Retrieves the movable limbs
        UnityEngine.GameObject[] movableLimbs = new GameObject[sortedHingeJoints.Count];
        for (int i = 0; i < sortedHingeJoints.Count; i++)
        {
            movableLimbs[i] = ((HingeJoint) sortedHingeJoints[i]).gameObject;
        }

        return movableLimbs;
    }


    /**
     * Splits the string into float array where elements are divided by a space (' ')
     */
    public static float[] string2floatArray(string original)
    {
        string[] stringData = original.Split(' ');
        float[] floatArrayData = new float[stringData.Length];

        for (int i = 0; i < stringData.Length; i++)
        {
            floatArrayData[i] = (float) (double.Parse(stringData[i])); //, System.Globalization.NumberStyles.Float);
        }

        return floatArrayData;
    }

    /**
     * Recursively goes through the ancestors of the object to find if it has the specified parent.
     */
    public static bool isAncestor(Transform inQuestion, string name)
    {
        if (inQuestion.parent == null)
        {
            return false;
        }

        if (inQuestion.parent.name.Equals(name))
        {
            return true;
        }
        else
        {
            return isAncestor(inQuestion.parent, name);
        }
    }

    public static float rads2degrees(float radians)
    {
        return (float) (radians / Math.PI * 180);
    }
}

/*
 * A client class that connects to a TCP connection server to exchange information with external programs.
 */
public class ClientConnectionHandler
{
    // Connection related variables
    private TcpClient input_socket;
    private TcpClient output_socket;
    private string HOST;
    private int INPUT_PORT;
    private int OUTPUT_PORT;
    private int INPUT_BUFFER_SIZE = 1024;
    private int TIMEMOUT_MS = 30000;

    // Thread-related variables
    private bool running = true;
    private String input_buffer;
    private Thread input_thread;
    private object lock_object;

    public ClientConnectionHandler(string HOST, int INPUT_PORT, int OUPUT_PORT)
    {
        init(HOST, INPUT_PORT, OUPUT_PORT);
    }

    public ClientConnectionHandler()
    {
        init("127.0.0.1", 6000, 5000);
    }

    private void init(string HOST, int INPUT_PORT, int OUTPUT_PORT)
    {
        this.HOST = HOST;
        this.INPUT_PORT = INPUT_PORT;
        this.OUTPUT_PORT = OUTPUT_PORT;

        // Creates socket
        this.output_socket = new TcpClient();
        this.output_socket.Connect(HOST, OUTPUT_PORT);
        this.output_socket.SendTimeout = TIMEMOUT_MS;
        this.output_socket.Client.NoDelay = true;
        this.output_socket.Client.Blocking = true;
        this.output_socket.NoDelay = true;
        Thread.Sleep(500);

        this.input_socket = new TcpClient();
        this.input_socket.Connect(HOST, INPUT_PORT);
        this.input_socket.ReceiveTimeout = TIMEMOUT_MS;
        this.input_socket.Client.NoDelay = true;
        this.input_socket.Client.Blocking = true;
        this.input_socket.NoDelay = true;
        Thread.Sleep(500);

        this.input_buffer = "";
        this.lock_object = new object();

        // Creates and starts the input thread
        this.input_thread = new Thread(new ThreadStart(data_receiver_thread_method));
        this.input_thread.Start();
    }

    private void data_receiver_thread_method()
    {
        while (running)
        {
            UserComponent.controlled_print("Looking for Python input...");
            byte[] b = new byte[INPUT_BUFFER_SIZE];
            this.input_socket.GetStream().Read(b, 0, INPUT_BUFFER_SIZE);

            UserComponent.controlled_print("Converted: " + System.Text.Encoding.UTF8.GetString(b));
            String received_string = System.Text.Encoding.UTF8.GetString(b).Trim((char) 0); //.Trim('#');
            UserComponent.controlled_print("received_string: " + received_string);

            lock (this.lock_object)
            {
                this.input_buffer += received_string;

                UserComponent.controlled_print("Buffer: " + this.input_buffer);
            }

            Thread.Sleep(1);
        }
    }

    public void println(string message)
    {
        // string blanks = "";
        // for (int i = 0; i < INPUT_BUFFER_SIZE - message.Length; i++)
        // {
        //     blanks += "#";
        // }

        byte[] output_message = System.Text.Encoding.UTF8.GetBytes(message + "$"); //  + blanks);
        this.output_socket.GetStream().Write(output_message, 0, output_message.Length);
        Thread.Sleep(1);
    }

    public String readline()
    {
        UserComponent.controlled_print("About to read new line from buffer...");
        while (this.running)
        {
            lock (this.lock_object)
            {
                UserComponent.controlled_print("Waiting for input in buffer... len=" +
                                               input_buffer.Length + " " + input_buffer);

                if (this.input_buffer.Length > 0)
                {
                    String[] message_list = input_buffer.TrimStart(' ').TrimStart('$').TrimStart(' ').Split('$');

                    if (message_list.Length > 1)
                    {
                        String message = message_list[0];
                        input_buffer = input_buffer.TrimStart(' ').TrimStart('$').TrimStart(' ')
                            .Substring(message.Length + 1);
                        UserComponent.controlled_print("Returning message: " + message);
                        return message;
                    }
                }
            }

            Thread.Sleep(1);
        }

        return null;
    }

    public void stop()
    {
        this.running = false;
        this.input_thread.Abort();
        input_socket.Close();
        output_socket.Close();
    }
}