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

public class FrameViewer : MonoBehaviour
{
    // Game variables
    private UnityEngine.GameObject[] movableLimbs;
    private UnityEngine.Rigidbody[] rigidBodies;
    public static bool freeze = false;
    public static bool earlyReset = false;

    void Start()
    {
        // Object fetching related logic
        while (GeneralData.isReady() == false)
        {
        }

        movableLimbs = GeneralData.getHingeLimbs("Hand Reference");
        rigidBodies = new Rigidbody[movableLimbs.Length];

        for (int i = 0; i < movableLimbs.Length; i++)
        {
            rigidBodies[i] = ((GameObject) movableLimbs[i]).GetComponent(typeof(Rigidbody)) as Rigidbody;
        }
    }


    /*
     * FixedUpdates refreshes 50 times per second by default
     */
    private void FixedUpdate()
    {
        if (freeze == false)
        {
            for (int i = 0; i < movableLimbs.Length; i++)
            {
                movableLimbs[i].transform.localRotation =
                    Quaternion.Euler(UserComponent.expectedAngles[i] * 57.29577951f, 0, 0);
            }
        }
    }

    /*
     * Refreshes as fast as the frame rate.
     */
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.F))
        {
            if (freeze == true)
            {
                freeze = false;
            }
            else
            {
                freeze = true;
            }
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            if (earlyReset == true)
            {
                earlyReset = false;
            }
            else
            {
                earlyReset = true;
            }
        }
    }


    private void OnDestroy()
    {
    }
}