using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.Linq;
using System.Threading;

public class UserComponent : MonoBehaviour
{
    // Game variables
    private UnityEngine.GameObject[] movableLimbs;
    private UnityEngine.Rigidbody[] rigidBodies;
    private float[] startingAngles;
    private float[] torquesToApply;

    // Process/training variables
    private bool training = true; // To be hard-coded (for now)
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
        process = new Process();

        // Calls python training script.
        process.StartInfo.FileName = @"C:\Users\Michael\AppData\Local\Microsoft\WindowsApps\python.exe";
        string scriptPath = @"C:\Git\Virtual-Hand\PythonScripts\ModelTrainerV3.py";
        // process.StartInfo.FileName = "\\..\\..\\PythonScripts\\ModelTrainer.py";
        string dataSetName = "RealData15"; // To be hard-coded (for now)
        string modelName = "FirstModelTest";
        // process.StartInfo.Arguments = scriptPath + " " + dataSetName + " " + modelName;
        process.StartInfo.Arguments = "-u "+ scriptPath;


        // Starts the process
        print(process.StartInfo.FileName);
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.StartInfo.RedirectStandardInput = true;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.Start();

        process.StandardInput.WriteLine(dataSetName);
        process.StandardInput.WriteLine(modelName);

        print("Started the Python process. ");

        // Interactions with the Python Script

        string acknowledgement = stdoutReadLine();
        print("Acknowledgement from Python: " + acknowledgement);
        if (acknowledgement.Equals("Ready") == false)
        {
            Console.Error.Write("Did not receive acknowledgement from Python script.");
            Quit();
        }
        else
        {
            // Obtains starting angles from the python script
            string[] stringBaseAngles = stdoutReadLine().Split(' ');
            for (int i = 0; i < stringBaseAngles.Length; i++)
            {
                startingAngles[i] = float.Parse(stringBaseAngles[i]);
            }

            print("Python angles obtained: " + stringBaseAngles.ToString());
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
                nextFrameTimeMs = long.Parse(stdoutReadLine());
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
                    process.StandardInput.WriteLine(getMilisecond() - sequenceStartTimeMs);
                    process.StandardInput.Flush();
                    // Step Loop-4 (as per protocol)
                    process.StandardInput.WriteLine(toSend);
                    process.StandardInput.Flush();

                    // Step Loop-5 (as per protocol)
                    string nextCommand = stdoutReadLine();
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
                        string[] stringTorques = stdoutReadLine().Split(' ');
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
        process.StandardInput.WriteLine("Ready");
        process.StandardInput.Flush();
    }

    private void Quit()
    {
        running = false; // Stops the thread loop
        // print("Stopping the script in 10 seconds...");
        // System.Threading.Thread.Sleep(10000); // Sleeps for 10 seconds
        process.StandardInput.Close();
        process.StandardOutput.Close();
        process.Close();
        print("Stopped.");
        return;
    }

    private string stdoutReadLine()
    {
        while (process.StandardOutput.Peek() == -1)
        {
            print("Waiting... for input from python.");
            System.Threading.Thread.Sleep(5);
        }

        return process.StandardOutput.ReadLine();
    }

    private void OnDestroy()
    {
        Quit();
    }
}