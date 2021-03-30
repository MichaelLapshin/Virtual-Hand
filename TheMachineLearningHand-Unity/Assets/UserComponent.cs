using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Net.Sockets;

public class UserComponent : MonoBehaviour
{
    // Game variables
    private UnityEngine.GameObject[] movableLimbs;
    private UnityEngine.Rigidbody[] rigidBodies;
    private float[] startingAngles;
    private float[] torquesToApply;

    // Process/training variables
    private bool training = true; // To be hard-coded (for now)
    private ClientConnectionHandler connection;
    private Process process;
    private Thread thread;
    private bool running = true;
    private bool requestReset = false;

    // Hands-on training variables
    private int resetCount = 0;
    private long sequenceStartTimeMs;
    private long nextFrameTimeMs = 0;
    private bool waitingForNewFrame;

    // Start is called before the first frame update
    void Start()
    {
        // Object fetching related logic
        HingeJoint[]
            hingeObjects =
                FindObjectsOfType(
                    typeof(HingeJoint)) as HingeJoint[]; // Should return all the finger limbs (since they have joints)
        var sortedHingeJoints = hingeObjects.OrderBy(go => go.name).ToList();

        movableLimbs = new GameObject[sortedHingeJoints.Count];
        rigidBodies = new Rigidbody[movableLimbs.Length];
        for (int i = 0; i < hingeObjects.Length; i++)
        {
            movableLimbs[i] = sortedHingeJoints[i].gameObject;
            rigidBodies[i] = ((GameObject) movableLimbs[i]).GetComponent(typeof(Rigidbody)) as Rigidbody;
        }

        startingAngles = new float[movableLimbs.Length];
        torquesToApply = new float[movableLimbs.Length];

        // Thread instantiation
        if (training == true)
        {
            thread = new Thread(this.RunTrainingThread);
            thread.Start();
        }
    }

    private void RunTrainingThread()
    {
        /*
         * Starting Sequence
         */
        // Model related logic

        connection = new ClientConnectionHandler();
        process = new Process();

        // Calls python training script.
        process.StartInfo.FileName = @"C:\Users\Michael\AppData\Local\Microsoft\WindowsApps\python.exe";
        string scriptPath = @"C:\Git\Virtual-Hand\PythonScripts\ModelTrainerV3.py";
        string dataSetName = "RealData15"; // To be hard-coded (for now)
        string modelName = "FirstModelTest";
        // process.StartInfo.Arguments = scriptPath + " " + dataSetName + " " + modelName;
        process.StartInfo.Arguments = scriptPath;


        // Starts the process
        print("Starting the process: " + process.StartInfo.FileName);
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.StartInfo.RedirectStandardInput = true;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.Start();
        System.Threading.Thread.Sleep(5000);

        // process.StandardInput.WriteLine(dataSetName);
        // process.StandardInput.WriteLine(modelName);
        // process.StandardInput.Flush();
        // connection.println("Received");
        connection.println(dataSetName);
        connection.println(modelName);


        print("Started the Python process. ");

        // Interactions with the Python Script

        // string acknowledgement = stdoutReadLine();
        string acknowledgement = connection.readline();
        print("Acknowledgement from Python: " + acknowledgement);
        if (acknowledgement.Equals("Ready") == false)
        {
            print("Did not receive acknowledgement from Python script.");
            Quit();
        }
        else
        {
            // Obtains starting angles from the python script
            print("Reading start angles...");
            string[] stringBaseAngles = connection.readline().Split(' ');
            for (int i = 0; i < stringBaseAngles.Length; i++)
            {
                startingAngles[i] = float.Parse(stringBaseAngles[i]);
            }

            print("Expecting start angles...");
            print("Python angles obtained: " + stringBaseAngles.ToString());
            // for (int i = 0; i < stringBaseAngles.Length; i++)
            // {
            // print(stringBaseAngles[i]);
            // }
            // print("Python angles obtained length: " + stringBaseAngles.Length);
            waitingForNewFrame = true;
            // ResetTrainingSequence_forThread();
            // System.Threading.Thread.Sleep(5000);
            Ready();
            // process.StandardInput.WriteLine("Ready");
        }
        // print("ready apparently sent");
        // print("RECEIVED " + stdoutReadLine());

        /*
         * Process Loop
         */
        while (running == true)
        {
            // Step Loop-0 (as per Pprotocol)
            if (waitingForNewFrame == true)
            {
                print("Next will be: ");
                // nextFrameTimeMs = long.Parse(stdoutReadLine());
                nextFrameTimeMs = long.Parse(connection.readline());
                print("Next time frame: " + nextFrameTimeMs);
                waitingForNewFrame = false;
            }
            else
            {
                // Step Loop-1 (as per protocol)
                if (getMilisecond() - sequenceStartTimeMs > nextFrameTimeMs)
                {
                    // Step Loop-2 (as per protocol)
                    // Composes the message to send to the python script
                    string toSend = "";
                    for (int i = 0; i < movableLimbs.Length; i++)
                    {
                        if (i != 0)
                        {
                            toSend += " ";
                        }

                        toSend += movableLimbs[i].transform.eulerAngles.x + " " + rigidBodies[i].angularVelocity.x;
                    }

                    // Sends data to python script
                    // Step Loop-3 (as per protocol)
                    // process.StandardInput.WriteLine(getMilisecond() - sequenceStartTimeMs);
                    // process.StandardInput.Flush();
                    connection.println((getMilisecond() - sequenceStartTimeMs).ToString());
                    // Step Loop-4 (as per protocol)
                    // process.StandardInput.WriteLine(toSend);
                    // process.StandardInput.Flush();
                    connection.println(toSend);

                    // Step Loop-5 (as per protocol)
                    // string nextCommand = stdoutReadLine();
                    string nextCommand = connection.readline();
                    // Step Loop-6 (as per protocol)
                    if (nextCommand.Equals("Reset"))
                    {
                        ResetTrainingSequence_forThread();

                        waitingForNewFrame = true;
                        Ready();
                    }
                    else if (nextCommand.Equals("Quit"))
                    {
                        Quit();
                    }
                    else if (nextCommand.Equals("Next"))
                    {
                        // Obtains and applies torques from python script to the limbs
                        // string[] stringTorques = stdoutReadLine().Split(' ');
                        string[] stringTorques = connection.readline().Split(' ');
                        for (int i = 0; i < movableLimbs.Length; i++)
                        {
                            torquesToApply[i] = float.Parse(stringTorques[i]);
                            // rigidBodies[i].AddTorque(new Vector3(float.Parse(stringTorques[i]), 0, 0), ForceMode.Force);
                        }

                        waitingForNewFrame = true;
                        Ready();
                    }
                    else
                    {
                        print(
                            "Unknown nextCommand sent from python script (" + nextCommand + "). Aborting program.");
                        Quit();
                    }
                }
            }
        }
    }

    private void FixedUpdate()
    {
        // Continuously applies existing torques to the limbs
        for (int i = 0; i < movableLimbs.Length; i++)
        {
            rigidBodies[i].AddTorque(new Vector3(torquesToApply[i], 0, 0), ForceMode.Force);
        }

        if (requestReset == true)
        {
            ResetTrainingSequence();
        }
    }


    private void ResetTrainingSequence_forThread()
    {
        requestReset = true;
        while (requestReset != false) // Waits until the main thread resets the hand
        {
            System.Threading.Thread.Sleep(1); // Sleeps for 1 ms while waiting
        }
    }

    private void ResetTrainingSequence()
    {
        resetCount++;
        print("The system is resetting. Reset #" + resetCount);
        for (int i = 0; i < movableLimbs.Length; i++)
        {
            ((GameObject) movableLimbs[i]).transform.eulerAngles.Set(startingAngles[i], 0, 0);
            rigidBodies[i].velocity = Vector3.zero;
            rigidBodies[i].angularVelocity = Vector3.zero;
            torquesToApply[i] = 0;
        }

        sequenceStartTimeMs = getMilisecond();
        requestReset = false;
    }

    // Returns current time in miliseconds
    private long getMilisecond()
    {
        return System.DateTime.Now.ToUniversalTime().Millisecond;
    }

    private void Ready()
    {
        // process.StandardInput.WriteLine("Ready");
        // process.StandardInput.Flush();
        connection.println("Ready");
    }

    private void Quit()
    {
        running = false; // Stops the thread loop
        // print("Stopping the script in 10 seconds...");
        // System.Threading.Thread.Sleep(10000); // Sleeps for 10 seconds
        process.StandardInput.Close();
        process.StandardOutput.Close();
        process.Close();
        connection.stop();
        print("Stopped.");
        return;
    }

    // private string stdoutReadLine()
    // {
    //     while (process.StandardOutput.Peek() == -1)
    //     {
    //         print("Waiting... for input from python.");
    //         System.Threading.Thread.Sleep(5);
    //     }
    //
    //     return process.StandardOutput.ReadLine();
    // }

    private void OnDestroy()
    {
        Quit();
    }
}

class ClientConnectionHandler
{
    private TcpClient socket;
    private Stream socketStream;
    private string HOST;
    private int PORT;
    private bool running = true;
    private string input_buffer = "";
    private Thread input_thread;

    public ClientConnectionHandler(string HOST, int PORT)
    {
        init(HOST, PORT);
    }

    public ClientConnectionHandler()
    {
        init("127.0.0.1", 5000);
    }

    private void init(string HOST, int PORT)
    {
        this.HOST = HOST;
        this.PORT = PORT;

        // Creates socket
        this.socket = new TcpClient();
        this.socket.Connect(HOST, PORT);
        this.socketStream = socket.GetStream();

        // Creates and starts the input thread
        this.input_thread = new Thread(this.data_receiver_thread_method);
        this.input_thread.Start();
    }

    public void data_receiver_thread_method()
    {
        while (running)
        {
            this.input_buffer += "" + socketStream.ReadByte().ToString();
        }
    }

    public void print(string message)
    {
        byte[] output_message = System.Text.Encoding.UTF8.GetBytes(message);
        this.socketStream.Write(output_message, 0, output_message.Length);
        this.socketStream.Flush();
    }

    public void println(string message)
    {
        byte[] output_message = System.Text.Encoding.UTF8.GetBytes(message + "\n");
        this.socketStream.Write(output_message, 0, output_message.Length);
        this.socketStream.Flush();
    }

    public string readline()
    {
        while (this.running)
        {
            string[] message_list = input_buffer.Split('\n');
            if (message_list.Length > 1)
            {
                string message = message_list[0];
                input_buffer = input_buffer.Substring(message.Length + 1);
                return message;
            }
        }

        return null;
    }

    public void stop()
    {
        this.running = false;
        socket.Close();
        this.socketStream.Close();
    }
}