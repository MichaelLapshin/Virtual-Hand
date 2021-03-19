using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;

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

        rigidBodies = new Rigidbody[movableLimbs.Length];
        for (int i = 0; i < movableLimbs.Length; i++)
        {
            movableLimbs[i] = hingeObjects[i].gameObject;
            rigidBodies[i] = ((GameObject) movableLimbs[i]).GetComponent(typeof(Rigidbody)) as Rigidbody;
        }

        startingAngles = new float[movableLimbs.Length];
        torquesToApply = new float[movableLimbs.Length];

        // Model related logic
        process = new Process();

        // Calls python training script.
        if (training == true)
        {
            process.StartInfo.FileName = "ModelTrainer.py";
            string dataSetName = "training_dataset_NAME.txt"; // To be hard-coded (for now)
            process.StartInfo.Arguments = dataSetName;
        }
        else
        {
            // string scriptName = "ModelReadingsMapper.py";
            // string modelName = ""; // Model to load in
            // process.StartInfo.FileName = scriptName + modelName;
        }

        // Starts the process
        process.Start();

        // Interactions with the Python Script
        if (training == true)
        {
            string acknowledgement = process.StandardOutput.ReadLine();
            if (acknowledgement.Equals("Ready") == false)
            {
                Console.Error.Write("Did not receive acknowledgement from Python script.");
                Quit();
            }
            else
            {
                // Obtains starting angles from the python script
                string[] stringBaseAngles = process.StandardOutput.ReadLine().Split(' ');
                for (int i = 0; i < stringBaseAngles.Length; i++)
                {
                    startingAngles[i] = float.Parse(stringBaseAngles[i]);
                }

                waitingForNewFrame = true;
                ResetTrainingSequence();
                Ready();
            }
        }
        else
        {
        }
    }

    private void FixedUpdate()
    {
        if (training == true)
        {
            // Step Loop-0 (as per Pprotocol)
            if (waitingForNewFrame == true)
            {
                nextFrameTimeMs = long.Parse(process.StandardOutput.ReadLine());
                waitingForNewFrame = false;
            }

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
                string nextCommand = process.StandardOutput.ReadLine();
                // Step Loop-6 (as per protocol)
                if (nextCommand.Equals("Reset"))
                {
                    ResetTrainingSequence();
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
                    string[] stringTorques = process.StandardOutput.ReadLine().Split(' ');
                    for (int i = 0; i < movableLimbs.Length; i++)
                    {
                        rigidBodies[i].AddTorque(new Vector3(float.Parse(stringTorques[i]), 0, 0), ForceMode.Force);
                    }

                    waitingForNewFrame = true;
                    Ready();
                }
                else
                {
                    Console.WriteLine(
                        "Unknown nextCommand sent from python script (" + nextCommand + "). Aborting program.");
                    Quit();
                }
            }
        }
        else
        {
            // Non-training code goes here
        }
    }

    private void ResetTrainingSequence()
    {
        resetCount++;
        Console.WriteLine("The system is resetting. Reset #" + resetCount);
        for (int i = 0; i < movableLimbs.Length; i++)
        {
            ((GameObject) movableLimbs[i]).transform.eulerAngles.Set(startingAngles[i], 0, 0);
            rigidBodies[i].velocity = Vector3.zero;
            rigidBodies[i].angularVelocity = Vector3.zero;
        }
        
        sequenceStartTimeMs = getMilisecond();
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
        Console.WriteLine("Stopping the script in 10 seconds...");
        System.Threading.Thread.Sleep(10000); // Sleeps for 10 seconds
        Console.WriteLine("Stopped.");
        this.enabled = false; // Kills this script.
        return;
    }
}